FROM python:3.11-slim

# Create non-root user
RUN groupadd -r medimind && useradd -r -g medimind -d /app -s /bin/false medimind

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_service/ .

# Ensure ml_models are readable
RUN chown -R medimind:medimind /app

USER medimind

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
