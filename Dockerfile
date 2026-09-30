FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    DISABLE_NNPACK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-railway.txt ./requirements-railway.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-railway.txt

COPY . .

EXPOSE 8080
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --worker-class gthread --workers 1 --threads 4 --timeout 300 app:app"]
