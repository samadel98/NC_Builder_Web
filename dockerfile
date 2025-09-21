FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

# install QD_Builder (provides nc-builder)
WORKDIR /opt
RUN git clone -b core-shell --depth 1 https://github.com/nlesc-nano/QD_Builder
RUN pip install --no-cache-dir ./QD_Builder

# app
WORKDIR /app
COPY api.py index.html attach/ ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] pyyaml ase

# health + bind to $PORT handled by api.py and cmd
CMD uvicorn api:app --host 0.0.0.0 --port $PORT

