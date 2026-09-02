# Services

Everything runs locally. `tilt up` opens the UI at **http://localhost:10350**,
where each service can be started, stopped and restarted individually, and each
one also writes `logs/<name>.log` on disk.

Only **chat** starts on its own. Everything else is defined, visible in the UI,
and waits for a click — including the two that carry risk, because a service
nobody can see is a service nobody knows exists.

| Service | URL | Starts on `tilt up`? | What it is | Risk |
| --- | --- | --- | --- | --- |
| **chat** | http://127.0.0.1:8800 | **yes** | The rules assistant, serving the local 7B via MLX. Password-gated; loopback only. | Low — cannot be reached off this machine on its own. |
| **tunnel** | `https://<random>.trycloudflare.com` (printed to `logs/tunnel.log`) | no | `cloudflared`, publishing **chat** to the internet. Waits for chat to be up. | **Publishes to the internet.** Hostname changes every start. |
| **rubric** | http://127.0.0.1:8000 | no | The rubric/adjudication authoring form. Model-free — this is the half that also deploys to fly.io. | Low. Writes only `submissions.jsonl`. |
| **webui** | http://127.0.0.1:8765 | no | Research console: label, author positions, run scripts. | **Runs training and eval as subprocesses.** Fine on loopback, remote code execution anywhere else — never give it `--lan` on an untrusted network. |
| **tests** | — | no | All 14 test files, run once and exit. | None. |
| *(Tilt itself)* | http://localhost:10350 | — | Start/stop, logs, per-service status. | None. |

## Running

```bash
tilt up                          # chat starts; the rest wait in the UI
tilt up chat tests               # only these two
tilt up -- --chat-port 8801      # when 8800 is already bound
tilt down                        # stop everything
```

Ports are the **scripts' own defaults**, so `tilt up` and running a script by
hand agree about where a service lives. Override per-service with
`--chat-port`, `--rubric-port`, `--webui-port`.

## Before starting `chat`

```bash
export CHAT_PASSWORD='...'
export CHAT_SECRET_KEY="$(openssl rand -hex 32)"
```

`chat_auth` **fails closed**: without a password it refuses any non-loopback
bind, and it checks before the model loads so the failure arrives in a second
rather than after a 40-second load. Loopback with no password is allowed for
solo local use.

## Why none of this is containerised

`mlx-metal` publishes **macOS arm64 wheels only** (`macosx_14_0_arm64`,
`macosx_15_0_arm64`, `macosx_26_0_arm64`). There is no Linux wheel, and Docker
on macOS runs a Linux VM with no Metal passthrough — so a containerised chat
service could not install MLX at all, let alone reach the GPU. Section 21.149
hit the same wall with fly.io.

`deploy/Dockerfile` is unrelated and still correct: it builds the **model-free
rubric form** for a public host, which is a different job from running services
locally.

## Logs

| File | Written by |
| --- | --- |
| `logs/chat.log` | chat |
| `logs/tunnel.log` | tunnel — **the public URL is in here** |
| `logs/rubric.log` | rubric |
| `logs/webui.log` | webui |
| `logs/tests.log` | tests |

Appended, not truncated, so a restart does not erase what happened before it.
`logs/` is gitignored.
