FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml setup.py requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY README.md DATA_SOURCES.md ./

RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1" && \
    python -m pip install -e '.[test]'

CMD ["python", "scripts/run_experiment.py", "--smoke", "--epochs", "12"]
