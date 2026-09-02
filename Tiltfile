# Local service manager for this repo.
#
# WHY local_resource AND NOT CONTAINERS FOR THE MODEL SERVICES
#
# `mlx-metal` publishes macOS arm64 wheels ONLY -- macosx_14_0_arm64,
# macosx_15_0_arm64, macosx_26_0_arm64. There is no Linux wheel, and Docker on
# macOS runs a Linux VM with no Metal passthrough, so a containerised
# chat_server could not install mlx at all, let alone reach the GPU. Same
# constraint that decided Section 21.149 against fly.io, one level closer to
# home. `deploy/Dockerfile` still exists and is still correct: it builds the
# model-free rubric form for a PUBLIC host, a different job from running
# services locally.
#
#   tilt up            # UI at http://localhost:10350
#   tilt up chat       # just one
#   tilt down          # stop everything
#
# EVERY service is defined here and visible in the UI. Only `chat` starts
# automatically; the rest wait for a click. An earlier version hid `webui` and
# `tunnel` behind --with-* flags, which added friction without adding safety --
# `auto_init=False` already means "does not run at startup", and a resource
# nobody can see is a resource nobody knows exists. The real protection lives
# in the services: webui.py binds loopback unless given --lan, and the tunnel
# cannot start before chat is up.
#
# Ports are the SCRIPTS' OWN DEFAULTS, so `tilt up` and running a script by
# hand agree about where a service lives. Override with
# `tilt up -- --chat-port 8801` when something is already bound.

config.define_string('chat-port', args=False)
config.define_string('rubric-port', args=False)
config.define_string('webui-port', args=False)
cfg = config.parse()

CHAT_PORT = cfg.get('chat-port', '8800')      # scripts/chat_server.py default
RUBRIC_PORT = cfg.get('rubric-port', '8000')  # scripts/rubric_server.py default
WEBUI_PORT = cfg.get('webui-port', '8765')    # scripts/webui.py default

# The venv interpreter by path, NOT `. mlx_env/bin/activate &&`. Activation is
# a shell builtin, so `exec . mlx_env/bin/activate && ...` dies with
# "exec: .: not found" -- and `tilt alpha tiltfile-result` cannot catch that,
# because it parses the Tiltfile without executing a single command.
PY = 'mlx_env/bin/python'


# Each service tees to its own file as well as the Tilt UI. The UI buffer is
# capped and dies with the process; the file survives a restart and can be
# diffed afterwards.
def logged(cmd, name):
    return 'mkdir -p logs && %s 2>&1 | tee -a logs/%s.log' % (cmd, name)


# ------------------------------------------------------------ chat (local 7B) --
# Loopback only, always. Section 21.150: the tunnel is the sole way in, so the
# service is never on the LAN and never depends on the firewall being right.
# --secure-cookies because a tunnel puts an HTTPS hostname in front even though
# uvicorn speaks plain HTTP here -- the flag describes how the USER reaches the
# service, not how the socket is bound.
local_resource(
    'chat',
    serve_cmd=logged(
        '%s -u scripts/chat_server.py --host 127.0.0.1 --port %s --secure-cookies'
        % (PY, CHAT_PORT), 'chat'),
    deps=['scripts/chat_server.py', 'scripts/chat_common.py', 'scripts/chat_auth.py',
          'scripts/common.py'],
    readiness_probe=probe(period_secs=5,
                          http_get=http_get_action(port=int(CHAT_PORT), path='/api/health')),
    links=[link('http://127.0.0.1:' + CHAT_PORT, 'chat (loopback)')],
    labels=['serving'],
    auto_init=True,
)

# ------------------------------------------------------------------- tunnel --
# Starting this PUBLISHES the chat surface to the internet, so it waits for a
# click. resource_deps keeps it from pointing at a dead port, which would serve
# a Cloudflare error page under your own hostname. The URL is random per start
# and lands in logs/tunnel.log.
local_resource(
    'tunnel',
    serve_cmd=logged(
        ('cloudflared tunnel --url http://127.0.0.1:%s ' +
         '--http-host-header 127.0.0.1:%s') % (CHAT_PORT, CHAT_PORT), 'tunnel'),
    resource_deps=['chat'],
    labels=['public'],
    auto_init=False,
)

# ------------------------------------------------------- rubric form (local) --
local_resource(
    'rubric',
    serve_cmd=logged(
        ('%s -u scripts/rubric_server.py ' +
         '--tasks data/gold/worksheets/tasks.json ' +
         '--host 127.0.0.1 --port %s') % (PY, RUBRIC_PORT), 'rubric'),
    deps=['scripts/rubric_server.py', 'scripts/common.py'],
    links=[link('http://127.0.0.1:' + RUBRIC_PORT, 'rubric form')],
    labels=['authoring'],
    auto_init=False,
)

# ----------------------------------------------------------- research console --
# webui.py runs training and evaluation as SUBPROCESSES. That is fine behind
# loopback and is remote code execution anywhere else, which is why it is never
# given --lan here. Defined and visible so it is discoverable; started on a
# click like every other non-chat resource.
local_resource(
    'webui',
    serve_cmd=logged('%s -u scripts/webui.py --port %s' % (PY, WEBUI_PORT), 'webui'),
    deps=['scripts/webui.py'],
    links=[link('http://127.0.0.1:' + WEBUI_PORT, 'research console')],
    labels=['authoring'],
    auto_init=False,
)

# ------------------------------------------------------------------ checks ----
# On demand, not on every edit: test_eval alone is 508 assertions.
local_resource(
    'tests',
    cmd=logged(('for t in scripts/test_*.py scripts/gameplay/test_*.py; ' +
                'do echo "== $t"; %s "$t" || exit 1; done') % PY, 'tests'),
    labels=['checks'],
    auto_init=False,
    trigger_mode=TRIGGER_MODE_MANUAL,
)

print(('chat starts automatically on 127.0.0.1:%s; tunnel/rubric/webui/tests ' +
       'wait for a click. `tunnel` publishes to the internet.') % CHAT_PORT)
