#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EXAMPLES_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_CLIENTS="${1:-3}"
CLIENT_CONFIG="${2:-./resources/configs/flamby/ixi/client_1.yaml}"
SERVER_CONFIG="${3:-./resources/configs/flamby/ixi/server_sz3.yaml}"
CLIENT_ID_OFFSET="${CLIENT_ID_OFFSET:-0}"

ERROR_BOUNDS=("1e-3" "1e-2" "3e-2" "5e-2" "1e-1")

total_runs=0
failed_runs=0

echo "[sweep] examples dir: ${EXAMPLES_DIR}"
echo "[sweep] python: ${PYTHON_BIN}"
echo "[sweep] server_config: ${SERVER_CONFIG}"
echo "[sweep] client_config: ${CLIENT_CONFIG}"
echo "[sweep] num_clients: ${NUM_CLIENTS}"
echo "[sweep] client_id_offset: ${CLIENT_ID_OFFSET}"
echo "[sweep] error_bounds: ${ERROR_BOUNDS[*]}"
echo

if [[ ! -f "${SERVER_CONFIG}" ]]; then
  echo "[sweep] missing server config: ${SERVER_CONFIG}"
  exit 1
fi

if [[ ! -f "${CLIENT_CONFIG}" ]]; then
  echo "[sweep] missing client config: ${CLIENT_CONFIG}"
  exit 1
fi

for eb in "${ERROR_BOUNDS[@]}"; do
  total_runs=$((total_runs + 1))
  suffix="unet_sz3_eb_${eb}"
  suffix="${suffix//./p}"

  echo "[sweep] (${total_runs}) error_bound=${eb}, log_suffix=${suffix}"

  "${PYTHON_BIN}" ./experiments/run_unet_federated.py \
    --server_config "${SERVER_CONFIG}" \
    --client_config "${CLIENT_CONFIG}" \
    --num_clients "${NUM_CLIENTS}" \
    --client_id_offset "${CLIENT_ID_OFFSET}" \
    --error_bound "${eb}" \
    --log_suffix "${suffix}"
  status=$?

  if [[ ${status} -ne 0 ]]; then
    failed_runs=$((failed_runs + 1))
    echo "[sweep] failed (exit=${status}) @ error_bound=${eb}"
  else
    echo "[sweep] done @ error_bound=${eb}"
  fi
  echo
done

echo "[sweep] finished: total=${total_runs}, failed=${failed_runs}"
if [[ ${failed_runs} -ne 0 ]]; then
  exit 1
fi

