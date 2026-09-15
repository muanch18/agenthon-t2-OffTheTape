FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version="2.0" \
      qfbench2.track="forecasting" \
      qfbench2.verb="forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QFBENCH_NETWORK=restricted

# Public contract pins: toolkit v2.4.0 (package metadata 2.3.1) and Track 2
# scorer source commit 83c6dc0 / package 3.1.0. Nothing installs at runtime.
RUN pip install --no-cache-dir \
      numpy==2.1.3 pandas==2.2.3 pyarrow==18.1.0 jsonschema==4.23.0 \
      "qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.4.0.tar.gz#subdirectory=common" \
      "qfbench2-track-forecasting @ https://github.com/Agenthon-2026/track2-forecasting-public/archive/83c6dc036bec03862b78e093bf8804d98964dad5.tar.gz"

WORKDIR /work
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . \
 && useradd --create-home --uid 1000 runner

USER runner
CMD ["forecast", "--help"]
