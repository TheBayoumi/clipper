FROM python:3.12-slim

ARG CLIPPER_SOURCE_SHA
ENV CLIPPER_SOURCE_SHA=${CLIPPER_SOURCE_SHA} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -c "import os,re,pathlib; s=os.environ.get('CLIPPER_SOURCE_SHA','').lower(); assert re.fullmatch(r'[0-9a-f]{40}', s), 'CLIPPER_SOURCE_SHA must be a full git SHA'; pathlib.Path('/app/.clipper-source-sha').write_text(s+'\\n', encoding='utf-8')"
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --upgrade pip \
    && pip install ".[open-models]"

ENTRYPOINT ["clipper"]
