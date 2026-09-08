#!/usr/bin/env bash
# Configure the Ollama macOS app for the experiment (T-03) and restart it.
#
# The app inherits its environment from launchd, so a plain `export` in a shell
# does nothing; `launchctl setenv` is the documented way. It does not survive a
# reboot: run this script again after restarting the machine, before a run or
# eval block. Values and rationale: configs/models.yaml and docs/setup.md.
set -euo pipefail

launchctl setenv OLLAMA_NUM_PARALLEL 2        # two dialogues at once (sim run --parallel 2)
launchctl setenv OLLAMA_MAX_LOADED_MODELS 2   # agent and simulator stay loaded together
launchctl setenv OLLAMA_KEEP_ALIVE -1         # never unload a model between turns
launchctl setenv OLLAMA_CONTEXT_LENGTH 8192   # server default; the client sets num_ctx too

# Restart the app so `ollama serve` is spawned again with the new environment.
# AppleScript `quit` is ignored when the app runs hidden, hence signals.
pkill -x Ollama 2>/dev/null || true
pkill -f 'ollama serve' 2>/dev/null || true
for _ in $(seq 1 40); do
  if ! pgrep -x Ollama >/dev/null && ! pgrep -f 'ollama serve' >/dev/null; then break; fi
  sleep 0.25
done
pkill -9 -x Ollama 2>/dev/null || true
pkill -9 -f 'ollama serve' 2>/dev/null || true
open -a Ollama
curl -s --retry 30 --retry-delay 1 --retry-connrefused --retry-all-errors -m 2 \
  http://localhost:11434/api/version >/dev/null

# Report what the server actually loaded, and fail loudly on any mismatch.
config=$(grep -E 'OLLAMA_NUM_PARALLEL' "${HOME}/.ollama/logs/server.log" | tail -1 | tr ' ' '\n' \
  | grep -E '^OLLAMA_(NUM_PARALLEL|MAX_LOADED_MODELS|KEEP_ALIVE|CONTEXT_LENGTH):')
version=$(curl -s http://localhost:11434/api/version | sed 's/.*"version":"\([^"]*\)".*/\1/')
echo "Ollama ${version} restarted with:"
echo "${config}" | sed 's/^/  /'
for expected in OLLAMA_NUM_PARALLEL:2 OLLAMA_MAX_LOADED_MODELS:2 OLLAMA_CONTEXT_LENGTH:8192; do
  echo "${config}" | grep -qx "${expected}" \
    || { echo "ERROR: the server did not load ${expected}" >&2; exit 1; }
done
# -1 is stored as the maximum Go duration, so the log shows 2562047h..., not -1.
echo "${config}" | grep -qE '^OLLAMA_KEEP_ALIVE:(-|2562047h)' \
  || { echo "ERROR: the server did not load OLLAMA_KEEP_ALIVE=-1" >&2; exit 1; }
