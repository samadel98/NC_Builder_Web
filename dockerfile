# Use micromamba to reproduce the QD_Builder conda env
FROM mambaorg/micromamba:1.5.8

# System deps
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# App
WORKDIR /app
COPY api.py index.html attach/ ./

# Get QD_Builder (core-shell branch) and create env
RUN git clone -b core-shell --depth 1 https://github.com/nlesc-nano/QD_Builder /opt/QD_Builder && \
    micromamba create -y -n nc-build -f /opt/QD_Builder/environment.yml && \
    micromamba install -y -n nc-build fastapi uvicorn && \
    micromamba clean -a -y

# FastAPI app must listen on $PORT and 0.0.0.0
ENV MAMBA_DEFAULT_ENV=nc-build
ENV PATH=/opt/conda/envs/nc-build/bin:$PATH

# Health endpoint expected by Render
# (ensure api.py serves /health and /)
CMD ["bash","-lc","uvicorn api:app --host 0.0.0.0 --port $PORT"]

