#!/bin/bash
# Starts 3 concurrent node-annotator reviewer instances, each on its own
# GPU/port, plus a Cloudflare quick-tunnel per instance so each can be
# shared as a public HTTPS link. Everything runs inside the docker
# container in tmux sessions, so it survives this script (and your shell)
# exiting.
#
# Usage:
#   docker/start_annotators.sh                    # start (or reuse) all 3
#   docker/start_annotators.sh --restart          # kill + fully restart all 3
#   docker/start_annotators.sh --restart-port 8600  # restart ONLY this one
#                                                    # reviewer (annotator +
#                                                    # tunnel), leaving the
#                                                    # other two untouched
#   docker/start_annotators.sh --stop             # stop everything, no restart
#   docker/start_annotators.sh --stop-port 8600   # stop ONLY this one
#
# Run from bare metal (this script itself shells into the container via
# `docker exec`) -- see CLAUDE.md / annotation_tool/README.md for why
# code execution otherwise always happens inside the container.
set -euo pipefail

CONTAINER_NAME="img-2-svg-pretraining-singlenode-venkat.kesav"
VENV_ACTIVATE="/environments/img_2_svg_pretraining/bin/activate"
APP_PATH="/code/src/img_2_svg_pretraining/annotation_tool/app.py"

# port : gpu index -- edit here to change the reviewer count/mapping.
# GPU 0 avoided: consistently the most loaded GPU on this shared host.
PORTS=(8600 8601 8602)
GPUS=(1 2 3)

_dexec() { docker exec "$CONTAINER_NAME" bash -c "$1"; }

_ensure_cloudflared() {
  _dexec "which cloudflared >/dev/null 2>&1 || (
    curl -L -o /usr/local/bin/cloudflared --silent --show-error \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 &&
    chmod +x /usr/local/bin/cloudflared)"
}

_stop_port() {
  local port="$1"
  echo "[port ${port}] stopping annotator + tunnel..."
  _dexec "tmux kill-session -t annotator_${port} 2>/dev/null || true"
  _dexec "tmux kill-session -t tunnel_${port} 2>/dev/null || true"
}

_stop_all() {
  echo "Stopping all annotator + tunnel tmux sessions..."
  for port in "${PORTS[@]}"; do
    _stop_port "$port"
  done
  echo "Stopped."
}

_port_index() {
  local target="$1"
  for i in "${!PORTS[@]}"; do
    [[ "${PORTS[$i]}" == "$target" ]] && { echo "$i"; return 0; }
  done
  return 1
}

if [[ "${1:-}" == "--stop" ]]; then
  _stop_all
  exit 0
fi

if [[ "${1:-}" == "--stop-port" ]]; then
  port="${2:?usage: docker/start_annotators.sh --stop-port <port>}"
  idx=$(_port_index "$port") || { echo "ERROR: ${port} is not a managed port (${PORTS[*]})." >&2; exit 1; }
  _stop_port "$port"
  exit 0
fi

if [[ "${1:-}" == "--restart" ]]; then
  _stop_all
  sleep 2
fi

if [[ "${1:-}" == "--restart-port" ]]; then
  port="${2:?usage: docker/start_annotators.sh --restart-port <port>}"
  idx=$(_port_index "$port") || { echo "ERROR: ${port} is not a managed port (${PORTS[*]})." >&2; exit 1; }
  _stop_port "$port"
  sleep 2
  # Narrow PORTS/GPUS to just this one port so every loop below (start,
  # wait, report) naturally only ever touches this reviewer.
  PORTS=("${PORTS[$idx]}")
  GPUS=("${GPUS[$idx]}")
fi

echo "=== Starting node annotator reviewer instances ==="

if ! docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" | grep -q "$CONTAINER_NAME"; then
  echo "ERROR: container ${CONTAINER_NAME} is not running. Run docker/init.sh first." >&2
  exit 1
fi

_ensure_cloudflared

for i in "${!PORTS[@]}"; do
  port="${PORTS[$i]}"
  gpu="${GPUS[$i]}"

  # Reuse an already-running instance/tunnel instead of double-starting --
  # tmux session names are the source of truth for "is this one up".
  if _dexec "tmux has-session -t annotator_${port} 2>/dev/null"; then
    echo "[port ${port}] annotator already running, leaving it alone (use --restart to force)."
  else
    echo "[port ${port}] starting annotator on GPU ${gpu}..."
    _dexec "tmux new-session -d -s annotator_${port} \
      'source ${VENV_ACTIVATE} && cd /code && CUDA_VISIBLE_DEVICES=${gpu} \
       streamlit run ${APP_PATH} --server.port ${port} --server.address 0.0.0.0 \
       --server.headless true 2>&1 | tee /tmp/annotator_${port}.log'"
  fi

  if _dexec "tmux has-session -t tunnel_${port} 2>/dev/null"; then
    echo "[port ${port}] tunnel already running, leaving it alone (use --restart to force)."
  else
    echo "[port ${port}] starting Cloudflare tunnel..."
    _dexec "rm -f /tmp/cf_tunnel_${port}.log"
    _dexec "tmux new-session -d -s tunnel_${port} \
      'cloudflared tunnel --url http://127.0.0.1:${port} 2>&1 | tee /tmp/cf_tunnel_${port}.log'"
  fi
done

echo ""
echo "Waiting for annotators to come up and tunnels to mint URLs..."

declare -a RESULT_PORT RESULT_GPU RESULT_STATUS RESULT_URL

for i in "${!PORTS[@]}"; do
  port="${PORTS[$i]}"
  gpu="${GPUS[$i]}"

  app_up="down"
  for _ in $(seq 1 30); do
    if _dexec "curl -s -m 2 -o /dev/null -w '%{http_code}' http://localhost:${port}" | grep -q 200; then
      app_up="up"
      break
    fi
    sleep 2
  done

  url=""
  for _ in $(seq 1 30); do
    url=$(_dexec "grep -oE 'https://[a-zA-Z0-9-]+\\.trycloudflare\\.com' /tmp/cf_tunnel_${port}.log 2>/dev/null | head -1" || true)
    [[ -n "$url" ]] && break
    sleep 2
  done

  RESULT_PORT+=("$port")
  RESULT_GPU+=("$gpu")
  RESULT_STATUS+=("$app_up")
  RESULT_URL+=("${url:-<no URL yet - check /tmp/cf_tunnel_${port}.log in container>}")
done

echo ""
echo "=== Reviewer links (share these directly) ==="
for i in "${!RESULT_PORT[@]}"; do
  printf "  Reviewer %d  (GPU %s, port %s, app %s):  %s\n" \
    "$((i + 1))" "${RESULT_GPU[$i]}" "${RESULT_PORT[$i]}" "${RESULT_STATUS[$i]}" "${RESULT_URL[$i]}"
done
echo ""
echo "Each link is a fresh Cloudflare quick-tunnel: no auth in front of it,"
echo "and it stops working if its tunnel is killed or the container restarts."
echo "Stop everything with: docker/start_annotators.sh --stop"
