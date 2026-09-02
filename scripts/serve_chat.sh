#!/usr/bin/env bash
# Serve the rules chat surface publicly, WITHOUT the model leaving this Mac.
#
# Section 21.149 rejected every off-machine hosting plan for one reason: mlx is
# Apple-Silicon-only, so anywhere else means serving a different model, and
# every number in Sections 21.139-21.148 describes THIS 7B. The model stays
# here; only the connection travels.
#
# cloudflared dials OUT to Cloudflare and Cloudflare proxies inbound requests
# back down that connection. So: no ports opened on the router, no static IP,
# no inbound firewall rule, and the public hostname terminates TLS at
# Cloudflare. The server itself still binds loopback only.
#
#   ./scripts/serve_chat.sh
#
# CHAT_PASSWORD and CHAT_SECRET_KEY must already be exported. chat_auth fails
# closed without them, but this script checks first so the failure arrives in
# one second rather than after a 40-second model load.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8800}"

if [[ -z "${CHAT_PASSWORD:-}" ]]; then
  cat >&2 <<'MSG'
CHAT_PASSWORD is not set. The tunnel would publish an unauthenticated page.

  export CHAT_PASSWORD='...'
  export CHAT_SECRET_KEY="$(openssl rand -hex 32)"

Keep them out of shell history (leading space, or a file you source).
MSG
  exit 2
fi

if [[ -z "${CHAT_SECRET_KEY:-}" ]]; then
  echo "note: CHAT_SECRET_KEY unset — sessions are signed with a random key," >&2
  echo "      so everyone is signed out whenever this restarts." >&2
fi

# Loopback ONLY. The tunnel is the sole way in, so the service is never exposed
# to the LAN and never depends on the machine's own firewall being right.
source mlx_env/bin/activate
# --secure-cookies because the PUBLIC hostname is HTTPS even though this
# process only speaks plain HTTP on loopback. Without it the session cookie
# has no Secure flag and a browser will send it over any future non-TLS
# path to the same host. The flag describes how the USER reaches the
# service, not how uvicorn is bound — that distinction is exactly what a
# reverse proxy makes easy to get wrong.
python -u scripts/chat_server.py --host 127.0.0.1 --port "$PORT" --secure-cookies &
SERVER_PID=$!
# Kill the tunnel too if the server dies, and vice versa: a tunnel pointing at
# a dead port serves a Cloudflare error page under YOUR hostname, which reads
# like the service is broken rather than stopped.
trap 'kill $SERVER_PID 2>/dev/null || true; kill ${TUNNEL_PID:-0} 2>/dev/null || true' EXIT INT TERM

until curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "server exited before becoming ready" >&2; exit 1; }
  sleep 2
done
echo "local server ready on 127.0.0.1:${PORT}"

# --http-host-header keeps the Host header pointing at the loopback origin;
# without it uvicorn sees the public hostname and some redirects break.
cloudflared tunnel --url "http://127.0.0.1:${PORT}" \
  --http-host-header "127.0.0.1:${PORT}" 2>&1 | tee /tmp/cloudflared.log &
TUNNEL_PID=$!

echo
echo "Watching for the public URL (also in /tmp/cloudflared.log) ..."
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -1 || true)
  [[ -n "$URL" ]] && { echo; echo "  PUBLIC URL:  $URL"; echo; break; }
  sleep 2
done

wait $SERVER_PID
