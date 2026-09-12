# Deploying the rubric form

A small web form so people who know Magic — and don't want to install anything —
can write rubrics from wherever they are.

## The boundary, first

**`webui.py` must never be deployed.** Its script runner executes training and
evaluation on the host and its store writes `gold_questions.jsonl` directly.
Those are fine behind a LAN token and are remote code execution on a public URL.

`rubric_server.py` is a separate program that shares no state with it:

| | Local console (`webui.py`) | Rubric form (`rubric_server.py`) |
| --- | --- | --- |
| Runs scripts | yes, allowlisted | **no** |
| Writes the gold set | yes | **no** |
| Reads corpora / models | yes | **no** |
| Dependencies | the full pip freeze | fastapi + uvicorn |
| Safe to expose | LAN only | yes, with a token |

The form reads one exported file and appends to one submissions log. Getting a
rubric into the gold set is a **local, reviewed step** — which is what keeps
"gold" meaning *a person looked at this*, rather than *someone typed it into a
box on the internet*.

The server validates inputs independently of the browser: authors are required
for attribution, rubric fields have bounded item and text sizes, adjudication
error numbers must be integers within the task rubric, boolean controls must be
JSON booleans, and a note is required when errors or `not_covered` are selected.
Invalid submissions are rejected before they reach the append-only log.

## Deploy

```bash
# 1. Export the questions to work on. Ships inside the image.
python scripts/author_rubrics.py --export-tasks
python scripts/author_rubrics.py --export-tasks --include-new 40   # also grow the set

# 2. Rename the app in deploy/fly.toml (fly names are globally unique), then:
fly launch --no-deploy --copy-config --config deploy/fly.toml
fly volumes create rubric_data --size 1

# 3. Set the access token. Everyone shares this one; it gates the form, and
#    contributors identify themselves by typing a name, not by logging in.
fly secrets set RUBRIC_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(12))')"

# 3b. Set the EXPORT credential. Separate from the one above on purpose: the
#     contributor token is shared by everyone who has the link, and the
#     submission log holds every other contributor's work. Without this,
#     GET /api/export is refused outright rather than falling back.
fly secrets set RUBRIC_EXPORT_TOKEN="$(openssl rand -hex 32)"

# 4. Ship it.
fly deploy --config deploy/fly.toml --dockerfile deploy/Dockerfile
```

Send each contributor `https://<your-app>.fly.dev/?t=<token>`. The token is
stored as a cookie for 30 days, so they follow the link once.

## Collect the work

```bash
curl -H "x-export-token: $RUBRIC_EXPORT_TOKEN" https://<your-app>.fly.dev/api/export > submissions.jsonl

python scripts/author_rubrics.py --ingest-submissions submissions.jsonl --dry-run
python scripts/author_rubrics.py --ingest-submissions submissions.jsonl
python scripts/validate_gold.py --to-eval
```

`--dry-run` prints every key point and error, flags anything with fewer than 2
key points or no common errors, and runs the "lead with the mistake" lint. Read
it before applying — ingest overwrites rubrics in place and there is no undo.

Attribution rides on each submission, so one file holds several authors and
each lands in its own `rubric_source`. That is what makes the per-author
breakdown in `eval.py --compare` possible.

## Updating the questions

Re-export and redeploy. Submissions live on the volume, not in the image, so
they survive:

```bash
python scripts/author_rubrics.py --export-tasks
fly deploy --config deploy/fly.toml --dockerfile deploy/Dockerfile
```

## Run it locally instead

No fly account needed — this is the same program:

```bash
python scripts/rubric_server.py --tasks data/gold/worksheets/tasks.json
python scripts/rubric_server.py --tasks tasks.json --host 0.0.0.0 --port 8080  # LAN, prints a token
```

A public `--host` without a token is refused rather than warned about.

## Notes

- **Cost.** 256MB shared-cpu-1x, suspends when idle. Comfortably inside fly's
  free allowance for a form a handful of people use.
- **Token scope.** One shared token gates the form; it is not per-user auth.
  Right for a handful of trusted people, wrong for anything public — treat the
  URL as the secret and don't post it anywhere. The link carries the token as
  `?t=`; the server exchanges it for a cookie and **redirects to the same URL
  without it**, so the secret does not sit in the address bar, the browser
  history or any `Referer` the page emits. The cookie is `HttpOnly`,
  `SameSite=Lax`, path `/`, and `Secure` whenever the request arrived over
  HTTPS (fly forwards `x-forwarded-proto`).
- **Export is a SEPARATE credential** (`RUBRIC_EXPORT_TOKEN`, sent as
  `x-export-token`). Contributors are trusted to use the form; that is not the
  same as being handed the complete submission log, which holds everyone
  else's rubrics, verdicts, notes and names. It matters for the measurement as
  well as the disclosure: `eval.py --compare` reports per-author agreement, and
  that is worth something only while reviewers are independent — 21.74 is the
  section about a reviewer who sees the answer key first. With a contributor
  token in force and no export token set, `/api/export` **fails closed**.
- **Submitted bodies are bounded** at the door — 256 KB per request, and the
  rubric form's own limits (`MAX_AUTHOR_LENGTH`, `MAX_RUBRIC_ITEMS`,
  `MAX_RUBRIC_ITEM_LENGTH`) reused for position and scenario drafts rather than
  a second set of numbers to keep in step.
- **`/healthz` is deliberately unauthenticated** so platform health checks pass.
  It returns a task count and nothing else. Gating it leaves every machine
  marked unhealthy and the app down.
- **Submissions are append-only.** Revisiting a question adds a row rather than
  replacing one; the import takes the latest per id, and the log stays an audit
  trail.
