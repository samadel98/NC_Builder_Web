# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8

# Create env with Python + heavy scientific deps (conda-forge = prebuilt binaries)
RUN micromamba create -y -n app -c conda-forge \
    python=3.11 \
    fastapi uvicorn python-multipart pydantic pyyaml \
    rdkit ase spglib numpy scipy networkx matplotlib \
 && micromamba clean -a -y

SHELL ["bash", "-lc"]
ENV MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app

# App code
COPY . /app

# Install QD_Builder from the 'core-shell' branch – this provides the 'nc-builder' CLI
RUN pip install --no-cache-dir "git+https://github.com/nlesc-nano/QD_Builder@core-shell"

# (Optional) sanity check – prints version at build-time
RUN python -c "import importlib,sys;print('QD_Builder OK');" && nc-builder --help >/dev/null 2>&1 || true

EXPOSE 8000
# Render sets $PORT in production; fall back to 8000 locally
CMD micromamba run -n app uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}

