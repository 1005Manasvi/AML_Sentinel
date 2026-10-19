# AML Sentinel demo image: API + dashboard on the bundled synthetic sample.
#
#   docker compose up --build      # then open http://localhost:8501
#
# Security: minimal slim base pinned by digest, hash-verified dependencies, no build
# tools or secrets in the final image, runs as an unprivileged user. Models are trained
# on the synthetic sample during the build, so the container starts without network
# access to model hubs and never sees real data.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

# ---------------------------------------------------------------- dependencies
FROM ${PYTHON_IMAGE} AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements/base.txt /tmp/base.txt
# --require-hashes: every wheel must match the lockfile digest, or the build fails.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes -r /tmp/base.txt

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE}
# libgomp1: OpenMP runtime LightGBM links against; nothing else is added.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin sentinel

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=dev \
    API_DATASET=Sample

COPY --from=deps /opt/venv /opt/venv
WORKDIR /app
# The source is installed via PYTHONPATH rather than `pip install .` so the build does
# not pull an unpinned build backend. Code is root-owned and read-only to the app user;
# only data/, models/ and reports/ (regenerated below) are writable.
COPY pyproject.toml LICENSE README.md ./
COPY .streamlit ./.streamlit
COPY src ./src
COPY --chown=sentinel:sentinel data/sample ./data/sample
RUN install -d -o sentinel -g sentinel models reports data/processed

USER sentinel
# Same seed-fixed steps as `make demo`, run once at build time.
RUN python -m aml_sentinel.cli generate-sample --seed 42 \
    && python -m aml_sentinel.cli prepare-data --dataset Sample \
    && python -m aml_sentinel.cli baseline --dataset Sample \
    && python -m aml_sentinel.cli train-tabular --dataset Sample

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
# 0.0.0.0 is required inside the container; compose publishes the ports on the host's
# loopback only. Keys are minted at start unless API_KEYS is passed in.
CMD ["python", "-m", "aml_sentinel.cli", "demo", "--skip-build", "--host", "0.0.0.0"]
