FROM python:3.12-slim
WORKDIR /workspace
COPY project/ .
RUN pip install -r requirements.txt --quiet 2>/dev/null || true
