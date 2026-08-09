FROM python:3.11-slim

# Create non-root user
RUN groupadd -r medimind && useradd -r -g medimind -d /app -s /bin/false medimind

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gosu libpq-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/django/ .
COPY docker/django-entrypoint.sh /usr/local/bin/django-entrypoint

# Create directories for static and media
RUN mkdir -p /app/staticfiles /app/media && \
    chown -R medimind:medimind /app && \
    chmod 755 /usr/local/bin/django-entrypoint

ENTRYPOINT ["/usr/local/bin/django-entrypoint"]

CMD ["gunicorn", "medimind.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "4"]
