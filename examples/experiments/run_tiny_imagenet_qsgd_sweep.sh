#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EXAMPLES_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_CLIENTS="${1:-10}"
CLIENT_CONFIG="${2:-./resources/configs/tiny_imagenet/client_1.yaml}"
SERVER_CONFIG="${3:-./resources/configs/tiny_imagenet/server_qsgd.yaml}"

# Prefer bit-based configuration. Default bits: 10, 7, 5, 4, 3.
# qsgd_level is computed as (2^bit - 1), which maps exactly to that bit width.
# Override examples:
# QSGD_BITS="10 7 5 4 3" bash ./experiments/run_tiny_imagenet_qsgd_sweep.sh
# QSGD_LEVELS="1023 127 31 15 7" bash ./experiments/run_tiny_imagenet_qsgd_sweep.sh
QSGD_BITS_STR="${QSGD_BITS:-10 7 5 4 3}"
QSGD_LEVELS_STR="${QSGD_LEVELS:-}"

declare -a QSGD_BITS
declare -a QSGD_LEVELS

if [[ -n "${QSGD_LEVELS_STR}" ]]; then
  read -r -a QSGD_LEVELS <<< "${QSGD_LEVELS_STR}"
else
  read -r -a QSGD_BITS <<< "${QSGD_BITS_STR}"
  for bit in "${QSGD_BITS[@]}"; do
    if [[ ! "${bit}" =~ ^[0-9]+$ ]] || [[ "${bit}" -le 0 ]]; then
      echo "[sweep] invalid bit value: ${bit}"
      exit 1
    fi
    level=$(( (1 << bit) - 1 ))
    QSGD_LEVELS+=("${level}")
  done
fi

if [[ ! -f "${SERVER_CONFIG}" ]]; then
  echo "[sweep] server config not found: ${SERVER_CONFIG}"
  exit 1
fi

total_runs=0
failed_runs=0

echo "[sweep] examples dir: ${EXAMPLES_DIR}"
echo "[sweep] python: ${PYTHON_BIN}"
echo "[sweep] num_clients: ${NUM_CLIENTS}"
echo "[sweep] client_config: ${CLIENT_CONFIG}"
echo "[sweep] server_config: ${SERVER_CONFIG}"
if [[ ${#QSGD_BITS[@]} -gt 0 ]]; then
  echo "[sweep] qsgd_bits: ${QSGD_BITS[*]}"
fi
echo "[sweep] qsgd_levels: ${QSGD_LEVELS[*]}"
echo

for idx in "${!QSGD_LEVELS[@]}"; do
  level="${QSGD_LEVELS[$idx]}"
  bit_label=""
  if [[ ${#QSGD_BITS[@]} -gt "${idx}" ]]; then
    bit_label=" (${QSGD_BITS[$idx]}bit)"
  fi
  total_runs=$((total_runs + 1))
  echo "[sweep] (${total_runs}) qsgd_level=${level}${bit_label}"

  "${PYTHON_BIN}" ./experiments/run_federated.py \
    --server_config "${SERVER_CONFIG}" \
    --client_config "${CLIENT_CONFIG}" \
    --num_clients "${NUM_CLIENTS}" \
    --qsgd_level "${level}"
  status=$?

  if [[ ${status} -ne 0 ]]; then
    failed_runs=$((failed_runs + 1))
    echo "[sweep] failed (exit=${status}) for qsgd_level=${level}"
  else
    echo "[sweep] done for qsgd_level=${level}"
  fi
  echo
done

echo "[sweep] finished: total=${total_runs}, failed=${failed_runs}"
if [[ ${failed_runs} -ne 0 ]]; then
  exit 1
fi
