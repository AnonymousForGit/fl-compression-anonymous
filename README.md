# Lossy Compression for FL

This repository focuses on lossy compression for federated learning and provides three implemented communication compressors:
- PredictorCompressor (magnitude + sign guided residual compression)
- SZ3 (error-bounded lossy compression)
- QSGD (quantized SGD)

The federated learning simulation and training workflow are built on top of APPFL.

## 1. Setup

### 1.1 Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 1.2 Install bundled compressor dependencies (required for `SZ3Compressor`)
Run:
```bash
bash src/appfl/compressor/install.sh
```

This installs SZ3 into:
- `~/.appfl/.compressor/SZ3/build/tools/sz3c/libSZ3c.so` (Linux)
- `~/.appfl/.compressor/SZ3/build/tools/sz3c/libSZ3c.dylib` (macOS)

`SZ3Compressor` uses that location by default.

### 1.3 Repo-local SZ3 build (used by PredictorCompressor default path)
PredictorCompressor in this repo searches for SZ3 under `SZ_NP/`.

Important:
- The local `SZ_NP` build is a customized SZ3 variant used for this project.
- It is a clean build with SZ3 built-in predictor path removed/disabled (non-predictor path).

Need to rebuild:
```bash
cd SZ_NP
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=.. ..
make -j
make install
```

Expected shared library location after build:
- `SZ_NP/lib64/libSZ3c.so` (Linux)
- `SZ_NP/lib64/libSZ3c.dylib` (macOS)

## 2. Compression Source Code

Core compressor implementations:
- `src/appfl/compressor/predictor_compressor.py`
- `src/appfl/compressor/sz3_compressor.py`
- `src/appfl/compressor/qsgd_compressor.py`

Compressor selection/registry:
- `src/appfl/compressor/compressor.py`
- `src/appfl/compressor/__init__.py`

## 3. Experiment Entrypoints

All runnable experiment scripts are in:
- `examples/experiments/`

Main runners:
- `examples/experiments/run_federated.py` (general image classification FL runs)
- `examples/experiments/run_unet_federated.py` (FLamby IXI U-Net runs)

## 4. Experiment Configs

Dataset configs are under:
- `examples/resources/configs/cifar10/`
- `examples/resources/configs/fashion_mnist/`
- `examples/resources/configs/caltech101/`
- `examples/resources/configs/tiny_imagenet/`
- `examples/resources/configs/flamby/ixi/`

Typical server config files:
- `server_predictor.yaml`
- `server_sz3.yaml` or `server_sz3_resnet18.yaml`
- `server_qsgd.yaml`

Client config files:
- `client_1.yaml` (template used for all clients in local experiments)

Note:
- Raw datasets are not bundled in this clean copy. Put datasets under `examples/datasets/RawData/` (and FLamby data under `data/`) before running experiments.

## 5. How To Run

### 5.1 Single run example (CIFAR10 + PredictorCompressor)
```bash
cd examples
python ./experiments/run_federated.py \
  --server_config ./resources/configs/cifar10/server_predictor.yaml \
  --client_config ./resources/configs/cifar10/client_1.yaml \
  --num_clients 2 \
  --error_bound 3e-2
```

### 5.2 Error-bound sweep (PredictorCompressor + SZ3)
```bash
cd examples
bash ./experiments/run_cifar10_predictor_sz3_sweep.sh
bash ./experiments/run_fashion_mnist_predictor_sz3_sweep.sh
bash ./experiments/run_caltech101_predictor_sz3_sweep.sh
bash ./experiments/run_tiny_imagenet_predictor_sz3_sweep.sh
```

### 5.3 QSGD bit sweep
```bash
cd examples
bash ./experiments/run_cifar10_qsgd_sweep.sh
bash ./experiments/run_fashion_mnist_qsgd_sweep.sh
bash ./experiments/run_caltech101_qsgd_sweep.sh
bash ./experiments/run_tiny_imagenet_qsgd_sweep.sh
```

Default QSGD bits in scripts: `10 7 5 4 3`.

### 5.4 FLamby IXI (U-Net)
```bash
cd examples
bash ./experiments/run_ixi_unet_predictor_eb_sweep.sh
bash ./experiments/run_ixi_unet_sz3_eb_sweep.sh
bash ./experiments/run_ixi_unet_qsgd_sweep.sh
```

## 6. Key Configuration Parameters

Common compression parameters (`client_configs.comm_configs.compressor_configs`):
- `enable_compression`: enable/disable compression.
- `lossy_compressor`: one of `PredictorCompressor`, `SZ3Compressor`, `QSGDCompressor`.
- `lossless_compressor`: backend for non-lossy path (`blosc`, `zstd`, etc.).
- `param_cutoff` / `param_count_threshold`: only compress tensors larger than this threshold.

PredictorCompressor-specific:
- `momentum_lr`: predictor update rate.
- `consistency_threshold`: sign-consistency threshold to decide predictor usage.
- `sz_config.error_bounding_mode`: SZ3 mode (`REL`, `ABS`, ...).
- `sz_config.error_bound`: target error bound.

SZ3-specific:
- `error_bounding_mode`: error control mode.
- `error_bound`: target error bound.

QSGD-specific:
- `qsgd_level`: quantization levels (`2^bits - 1` in provided sweep scripts).

Training/aggregation parameters:
- `train_configs.mode`: local update mode (`epoch` or `step`).
- `train_configs.num_local_epochs` / `train_configs.num_local_steps`: local compute budget.
- `train_configs.send_gradient`: send model delta/gradient instead of full model.
- `server_configs.aggregator_kwargs.expect_gradient`: must match `send_gradient`.

## 7. Outputs

Run logs and metrics are written to:
- `examples/output/`

The folder is created at runtime and intentionally not tracked in this clean repo copy.
