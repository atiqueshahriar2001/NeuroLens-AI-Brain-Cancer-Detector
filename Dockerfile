FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-railway.txt ./requirements-railway.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-railway.txt

COPY . .

EXPOSE 8080
CMD ["sh", "-c", "python -m waitress --listen=0.0.0.0:${PORT:-8080} --threads=${WAITRESS_THREADS:-4} app:app"]
