# Rubric authoring form. See deploy/README.md.
#
# Copies exactly three things: the server, common.py (which it imports for the
# `lead with the mistake` lint), and the exported task file. Not the gold set,
# not the candidates, not the corpora, not the models. If a future edit makes
# this COPY line longer, that is the moment to ask whether the thing being
# added should be reachable from a public URL.
FROM python:3.12-slim

WORKDIR /app

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/rubric_server.py scripts/
COPY scripts/common.py scripts/
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
