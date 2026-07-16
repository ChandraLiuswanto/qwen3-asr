#!/usr/bin/env bash
#
# Benchmark /v1/audio/transcriptions under concurrency, inside a conda env.
#
# USAGE
#   ./scripts/h100/bench.sh --audio /path/to/5min.wav
#   CONDA_ENV=myenv ./scripts/h100/bench.sh --audio clip.wav --levels 1,4,8,10
#
#   Every flag is passed through to bench_concurrency.py (--levels, --rounds,
#   --base-url, --no-diarization, --json-out, ...). See its --help.
#
# THE RUN THAT ACTUALLY ANSWERS THE QUESTION
#   One run tells you nothing on its own — you need a baseline to compare to.
#   The server reads VLLM_OFFLINE_CONCURRENCY at boot, so this script cannot
#   flip it for you. Do this:
#
#     1. Restart the service with VLLM_OFFLINE_CONCURRENCY=1   (serialized: the
#        old behavior, before the lock was narrowed)
#          ./scripts/h100/bench.sh --audio clip.wav --json-out baseline.json
#     2. Restart with the default (4)
#          ./scripts/h100/bench.sh --audio clip.wav --json-out concurrent.json
#     3. Compare. The delta IS what change A bought. If there is no delta,
#        change A bought correctness only — which is still real, but say so
#        rather than claiming a speedup nobody measured.
#
# ORDER OF OPERATIONS
#   Run scripts/h100/test_offline_mixing.py FIRST. It checks the transcripts are
#   correct under concurrency. Benchmarking a service that returns mixed-up text
#   measures how fast it produces wrong answers.
#
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-qwen3-asr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- locate conda ------------------------------------------------------------
# `conda activate` needs the shell function, which a non-interactive shell does
# not have. Source conda.sh; fall back to `conda info --base` if CONDA_EXE is
# not exported.
if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "ERROR: conda not found on PATH and CONDA_EXE is unset." >&2
  echo "       Set CONDA_EXE, or run this from a shell where conda is available." >&2
  exit 1
fi

# shellcheck disable=SC1091
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: $CONDA_BASE/etc/profile.d/conda.sh not found." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "ERROR: conda env '$CONDA_ENV' does not exist." >&2
  echo "       Available:" >&2
  conda env list | sed 's/^/         /' >&2
  echo "       Set CONDA_ENV=<name> to pick one." >&2
  exit 1
fi

conda activate "$CONDA_ENV"
echo "conda env : $CONDA_ENV  ($(python -V 2>&1))"
echo "python    : $(command -v python)"

# --- report the knobs, so two runs can be told apart -------------------------
# These are the CLIENT's view of the env. The server was configured at ITS boot;
# if you started it elsewhere, these may not reflect what it is actually running.
echo "client-side VLLM_OFFLINE_CONCURRENCY  : ${VLLM_OFFLINE_CONCURRENCY:-<unset, server default>}"
echo "client-side VLLM_WS_DECODE_CONCURRENCY: ${VLLM_WS_DECODE_CONCURRENCY:-<unset, server default>}"
echo "base url  : ${ASR_BASE_URL:-http://localhost:8000}"
echo

exec python "$REPO_ROOT/scripts/h100/bench_concurrency.py" "$@"
