# Rubric authoring / judge adjudication form. See deploy/README.md.
#
# Copies exactly four things: the server, common.py (which it imports for the
# `lead with the mistake` lint), and TWO exported task files, one per form. Not the gold set,
# not the candidates, not the corpora, not the models. If a future edit makes
# this COPY line longer, that is the moment to ask whether the thing being
# added should be reachable from a public URL.
#
# That question was asked when adjudication was added, and the answer was yes:
# the task file it serves is produced by `adjudicate.py --export-tasks` and is
# the same class of content as the rubric one — a question, an answer, and a
# numbered list of candidate errors. It carries no judge output, which the
# server re-checks at load rather than trusting, and it reaches nothing else.
# The COPY list did not grow: the two task kinds share one slot, and which form
# is served is decided by the file's own `kind` field.
#
# `webui.py` remains undeployable and is deliberately absent. Its script runner
# executes training and evaluation on the host and its store writes the gold set
# directly; both are fine behind a LAN token and are remote code execution on a
# public URL.
FROM python:3.12-slim

WORKDIR /app

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/rubric_server.py scripts/
COPY scripts/common.py scripts/
# TWO slots, one per form, so one deployment serves both. The kind of each file
# comes from the file itself, never from the filename or the order:
#   python scripts/adjudicate.py --export-tasks OUT              (adjudication)
#   python scripts/author_rubrics.py --export-tasks --include-new N  (rubric)
#
# The list grew from one to two on 2026-08-21, which is the moment this comment
# says to stop and ask whether the new thing belongs on a public URL. It does,
# and for the reason already given below: a rubric task is a question, a
# reference answer and a machine draft — the same class of content as an
# adjudication task, carrying no judge output and reaching nothing else. What is
# NOT here is unchanged: no gold set, no positions, no queue, no runs, no
# corpora, no adapters, no webui.py.
COPY data/gold/worksheets/tasks.json          data/gold/worksheets/
COPY data/gold/worksheets/rubric_tasks.json   data/gold/worksheets/
# Grew to three on 2026-08-29, same question asked again. `position_tasks.json`
# is the Section 21.126 export of recorded-game boards (kind: "position" per
# task): still a question, a machine-drafted board render, and (for these 71)
# no reference answer — no gold set, no candidates, no legal_actions solver.
# The 21.126 fix is what makes it safe to point a contributor at: before it,
# this file's tasks rendered through the rules-question card and asserted a
# verified answer that does not exist for a board.
COPY data/gold/worksheets/position_tasks.json data/gold/worksheets/

# Submissions land on a mounted volume so a redeploy does not discard work.
# TASKS_FILE/ALSO_TASKS_FILE here are the image's own defaults for a bare
# `docker run`; fly.toml's [env] overrides them for the actual deployment,
# since which two of the three files are live is a per-deploy decision, not a
# property of the image.
ENV TASKS_FILE=/app/data/gold/worksheets/tasks.json \
    ALSO_TASKS_FILE=/app/data/gold/worksheets/rubric_tasks.json \
    SUBMISSIONS_FILE=/data/submissions.jsonl \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

# No --no-token: the server refuses a public bind without auth, and RUBRIC_TOKEN
# is set as a fly secret.
CMD ["python", "scripts/rubric_server.py"]
