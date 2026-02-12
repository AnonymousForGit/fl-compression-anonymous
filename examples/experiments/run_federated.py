"""
Single-process federated learning runner.
"""

import argparse
import math
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent


def _to_file_token(value):
    token = str(value).strip().lower()
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in token)


def _error_bound_tag(value):
    if value is None:
        return "eb_na"
    try:
        return f"eb_{float(value):.0e}".replace("+", "")
    except (TypeError, ValueError):
        return f"eb_{_to_file_token(value)}"


def _qsgd_bits_tag(level):
    if level is None:
        return "bits_na"
    try:
        quant_levels = int(level)
        if quant_levels <= 0:
            return "bits_na"
        # QSGD stores levels in [0, s], so nominal bits are ceil(log2(s + 1)).
        bits = max(1, math.ceil(math.log2(quant_levels + 1)))
        return f"bits_{bits}"
    except (TypeError, ValueError):
        return f"bits_{_to_file_token(level)}"


def _compression_log_suffix(server_cfg):
    try:
        compressor_cfg = server_cfg.client_configs.comm_configs.compressor_configs
    except AttributeError:
        return "compression_unknown_eb_na"

    if not compressor_cfg.get("enable_compression", False):
        return "compression_none_eb_na"

    lossy = compressor_cfg.get("lossy_compressor", "unknown")
    lossy_token = _to_file_token(lossy)
    if "qsgd" in lossy_token:
        qsgd_level = compressor_cfg.get("qsgd_level", None)
        if qsgd_level is None:
            qsgd_level = compressor_cfg.get("quantization_levels", None)
        return f"compression_{lossy_token}_{_qsgd_bits_tag(qsgd_level)}"

    error_bound = None
    if "sz_config" in compressor_cfg and compressor_cfg.sz_config is not None:
        error_bound = compressor_cfg.sz_config.get("error_bound", None)
    if error_bound is None:
        error_bound = compressor_cfg.get("error_bound", None)

    return f"compression_{lossy_token}_{_error_bound_tag(error_bound)}"


def _append_suffix(base_name, suffix):
    base = str(base_name) if base_name is not None else "result"
    return f"{base}_{suffix}" if suffix not in base else base


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--server_config", type=str, default="./resources/configs/tiny_imagenet/server_predictor.yaml"
)
# argparser.add_argument(
#     "--server_config", type=str, default="./resources/configs/cifar10/server_qsgd.yaml"
# )
argparser.add_argument(
    "--client_config", type=str, default="./resources/configs/tiny_imagenet/client_1.yaml"
)
argparser.add_argument("--num_clients", type=int, default=10)
argparser.add_argument("--error_bound", type=float, default=1e-3)
argparser.add_argument("--qsgd_level", type=int, default=None)
args = argparser.parse_args()

# Load server agent configurations and set the number of clients
server_agent_config = OmegaConf.load(args.server_config)
server_agent_config.server_configs.num_clients = args.num_clients
compressor_cfg = server_agent_config.client_configs.comm_configs.compressor_configs
if "sz_config" in compressor_cfg and compressor_cfg.sz_config is not None:
    compressor_cfg.sz_config.error_bound = args.error_bound
elif "error_bound" in compressor_cfg:
    compressor_cfg.error_bound = args.error_bound
if args.qsgd_level is not None:
    compressor_cfg.qsgd_level = args.qsgd_level
    if "quantization_levels" in compressor_cfg:
        compressor_cfg.quantization_levels = args.qsgd_level
log_suffix = _compression_log_suffix(server_agent_config)
server_agent_config.server_configs.logging_output_filename = _append_suffix(
    server_agent_config.server_configs.get("logging_output_filename", "result"), log_suffix
)

# Create server agent
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Load base client configurations and set corresponding fields for different clients
client_agent_configs = [
    OmegaConf.load(args.client_config) for _ in range(args.num_clients)
]
for i in range(args.num_clients):
    client_agent_configs[i].client_id = f"Client{i + 1}"
    client_agent_configs[i].data_configs.dataset_kwargs.num_clients = args.num_clients
    client_agent_configs[i].data_configs.dataset_kwargs.client_id = i
    client_agent_configs[i].data_configs.dataset_kwargs.visualization = (
        True if i == 0 else False
    )
    client_agent_configs[i].train_configs.logging_output_filename = _append_suffix(
        client_agent_configs[i].train_configs.get("logging_output_filename", "result"),
        log_suffix,
    )
    # Enabling wandb for one client is sufficient in this runner.
    if hasattr(client_agent_configs[i], "wandb_configs") and client_agent_configs[
        i
    ].wandb_configs.get("enable_wandb", False):
        if i == 0:
            client_agent_configs[i].wandb_configs.enable_wandb = True
        else:
            client_agent_configs[i].wandb_configs.enable_wandb = False

# Load client agents
client_agents = [
    ClientAgent(client_agent_config=client_agent_configs[i])
    for i in range(args.num_clients)
]

# Get additional client configurations from the server
client_config_from_server = server_agent.get_client_configs()
for client_agent in client_agents:
    client_agent.load_config(client_config_from_server)

# Load initial global model from the server
init_global_model = server_agent.get_parameters(serial_run=True)
for client_agent in client_agents:
    client_agent.load_parameters(init_global_model)

# [Optional] Set number of local data to the server
for i in range(args.num_clients):
    sample_size = client_agents[i].get_sample_size()
    server_agent.set_sample_size(
        client_id=client_agents[i].get_id(), sample_size=sample_size
    )

while not server_agent.training_finished():
    new_global_models = []
    for client_agent in client_agents:
        # Client local training
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}
        # "Send" local model to server and get a Future object for the new global model
        # The Future object will be resolved when the server receives local models from all clients
        new_global_model_future = server_agent.global_update(
            client_id=client_agent.get_id(),
            local_model=local_model,
            blocking=False,
            **metadata,
        )
        new_global_models.append(new_global_model_future)
    # Load the new global model from the server
    for client_agent, new_global_model_future in zip(client_agents, new_global_models):
        client_agent.load_parameters(new_global_model_future.result())
