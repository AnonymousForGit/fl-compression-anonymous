#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EXAMPLES_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_CLIENTS="${1:-2}"
CLIENT_CONFIG="${2:-./resources/configs/tiny_imagenet/client_1.yaml}"

ERROR_BOUNDS=("1e-3" "1e-2" "3e-2" "5e-2" "1e-1")

declare -a SERVER_CONFIGS=(
  "./resources/configs/tiny_imagenet/server_predictor.yaml"
  "./resources/configs/tiny_imagenet/server_sz3.yaml"
)

total_runs=0
failed_runs=0

echo "[sweep] examples dir: ${EXAMPLES_DIR}"
echo "[sweep] python: ${PYTHON_BIN}"
echo "[sweep] num_clients: ${NUM_CLIENTS}"
echo "[sweep] client_config: ${CLIENT_CONFIG}"
echo "[sweep] error_bounds: ${ERROR_BOUNDS[*]}"
echo

for server_cfg in "${SERVER_CONFIGS[@]}"; do
  if [[ ! -f "${server_cfg}" ]]; then
    echo "[sweep] skip missing config: ${server_cfg}"
    continue
  fi

  for eb in "${ERROR_BOUNDS[@]}"; do
    total_runs=$((total_runs + 1))
    echo "[sweep] (${total_runs}) server_config=${server_cfg} error_bound=${eb}"

    "${PYTHON_BIN}" ./experiments/run_federated.py \
      --server_config "${server_cfg}" \
      --client_config "${CLIENT_CONFIG}" \
      --num_clients "${NUM_CLIENTS}" \
      --error_bound "${eb}"
    status=$?

    if [[ ${status} -ne 0 ]]; then
      failed_runs=$((failed_runs + 1))
      echo "[sweep] failed (exit=${status}) for ${server_cfg} @ error_bound=${eb}"
    else
      echo "[sweep] done for ${server_cfg} @ error_bound=${eb}"
    fi
    echo
  done
done

echo "[sweep] finished: total=${total_runs}, failed=${failed_runs}"
if [[ ${failed_runs} -ne 0 ]]; then
  exit 1
fi

