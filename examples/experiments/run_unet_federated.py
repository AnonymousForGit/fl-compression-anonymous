"""
Single-process runner for U-Net style experiments (for example, FLamby IXI).
"""

import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent


def _append_suffix(base_name: str, suffix: str) -> str:
    if not suffix:
        return base_name
    base = str(base_name) if base_name is not None else "result"
    return f"{base}_{suffix}" if suffix not in base else base


def _set_if_exists(cfg, key, value) -> None:
    if key in cfg:
        cfg[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_config",
        type=str,
        default="./resources/configs/flamby/ixi/server_fedavg.yaml",
    )
    parser.add_argument(
        "--client_config",
        type=str,
        default="./resources/configs/flamby/ixi/client_1.yaml",
    )
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument(
        "--client_id_offset",
        type=int,
        default=0,
        help="Offset for dataset_kwargs.client_id (default: 0).",
    )
    parser.add_argument(
        "--log_suffix",
        type=str,
        default="unet_run",
        help="Suffix appended to server/client log filenames.",
    )
    parser.add_argument(
        "--error_bound",
        type=float,
        default=None,
        help="Override compressor error bound when compression is enabled.",
    )
    parser.add_argument(
        "--qsgd_level",
        type=int,
        default=None,
        help="Override QSGD quantization level when QSGD compression is enabled.",
    )
    args = parser.parse_args()

    # Load server config and override number of clients.
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = args.num_clients

    # Optional compression error-bound override.
    if args.error_bound is not None:
        try:
            compressor_cfg = server_agent_config.client_configs.comm_configs.compressor_configs
            if "sz_config" in compressor_cfg and compressor_cfg.sz_config is not None:
                compressor_cfg.sz_config.error_bound = args.error_bound
            elif "error_bound" in compressor_cfg:
                compressor_cfg.error_bound = args.error_bound
        except AttributeError:
            pass

    # Optional QSGD-level override.
    if args.qsgd_level is not None:
        try:
            compressor_cfg = server_agent_config.client_configs.comm_configs.compressor_configs
            if "qsgd_level" in compressor_cfg:
                compressor_cfg.qsgd_level = args.qsgd_level
        except AttributeError:
            pass

    server_agent_config.server_configs.logging_output_filename = _append_suffix(
        server_agent_config.server_configs.get("logging_output_filename", "result"),
        args.log_suffix,
    )

    # Build client configs from one template.
    client_agent_configs = [
        OmegaConf.load(args.client_config) for _ in range(args.num_clients)
    ]
    for i in range(args.num_clients):
        cfg = client_agent_configs[i]
        cfg.client_id = f"Client{i + 1}"

        if "data_configs" in cfg and "dataset_kwargs" in cfg.data_configs:
            kwargs = cfg.data_configs.dataset_kwargs
            _set_if_exists(kwargs, "num_clients", args.num_clients)
            _set_if_exists(kwargs, "client_id", i + args.client_id_offset)
            # Keep compatibility with loaders that already support visualization.
            if "visualization" in kwargs:
                kwargs.visualization = i == 0

        cfg.train_configs.logging_output_filename = _append_suffix(
            cfg.train_configs.get("logging_output_filename", "result"),
            args.log_suffix,
        )

        # Enabling wandb on one client is usually sufficient.
        if (
            "wandb_configs" in cfg
            and cfg.wandb_configs.get("enable_wandb", False)
            and i > 0
        ):
            cfg.wandb_configs.enable_wandb = False

    # Create server/client agents.
    server_agent = ServerAgent(server_agent_config=server_agent_config)
    client_agents = [
        ClientAgent(client_agent_config=client_agent_configs[i])
        for i in range(args.num_clients)
    ]

    # Merge shared client configs from server.
    client_config_from_server = server_agent.get_client_configs()
    for client_agent in client_agents:
        client_agent.load_config(client_config_from_server)

    # Broadcast initial global model.
    init_global_model = server_agent.get_parameters(serial_run=True)
    for client_agent in client_agents:
        client_agent.load_parameters(init_global_model)

    # Optional: set per-client sample size for weighted aggregation.
    for client_agent in client_agents:
        server_agent.set_sample_size(
            client_id=client_agent.get_id(),
            sample_size=client_agent.get_sample_size(),
        )

    # Main FL loop.
    while not server_agent.training_finished():
        new_global_models = []
        for client_agent in client_agents:
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            new_global_model = server_agent.global_update(
                client_id=client_agent.get_id(),
                local_model=local_model,
                blocking=False,
                **metadata,
            )
            new_global_models.append(new_global_model)

        for client_agent, global_model_or_future in zip(client_agents, new_global_models):
            if hasattr(global_model_or_future, "result"):
                global_model_or_future = global_model_or_future.result()
            client_agent.load_parameters(global_model_or_future)


if __name__ == "__main__":
    main()
