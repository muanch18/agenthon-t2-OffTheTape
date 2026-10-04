FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version="2.0" \
      qfbench2.track="forecasting" \
      qfbench2.verb="forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QFBENCH_NETWORK=restricted

# Production only needs the numeric agent and Parquet writer. The official scorer
# lives in Dockerfile.verifier for rehearsal, outside the submitted image.
RUN pip install --no-cache-dir --no-compile \
      numpy==2.1.3 pandas==2.2.3 pyarrow==18.1.0

WORKDIR /work
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-compile --no-deps . \
 && useradd --create-home --uid 1000 runner

USER runner
CMD ["forecast", "--help"]
