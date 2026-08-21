# Rubric authoring / judge adjudication form. See deploy/README.md.
#
# Copies exactly three things: the server, common.py (which it imports for the
# `lead with the mistake` lint), and ONE exported task file. Not the gold set,
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
# One slot, either kind. Point TASKS_FILE at whichever was exported:
#   python scripts/author_rubrics.py --export-tasks      (rubric authoring)
#   python scripts/adjudicate.py --export-tasks OUT      (judge adjudication)
COPY data/gold/worksheets/tasks.json data/gold/worksheets/

# Submissions land on a mounted volume so a redeploy does not discard work.
ENV TASKS_FILE=/app/data/gold/worksheets/tasks.json \
    SUBMISSIONS_FILE=/data/submissions.jsonl \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

# No --no-token: the server refuses a public bind without auth, and RUBRIC_TOKEN
# is set as a fly secret.
CMD ["python", "scripts/rubric_server.py"]
