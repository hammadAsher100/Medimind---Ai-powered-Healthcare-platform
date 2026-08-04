FROM python:3.11-slim

RUN groupadd --system medimind \
  && useradd \
  --system \
  --gid medimind \
  --home-dir /app \
  --shell /usr/sbin/nologin \
  medimind

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_BASE_DIR=/opt/medimind/models
ENV CNN_PNEUMONIA_MODEL_PATH=/opt/medimind/models/pneumonia/resnet50_pneumonia.keras

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY ai_service/ .

RUN mkdir -p /opt/medimind/models \
  && chown -R medimind:medimind /app /opt/medimind

USER medimind

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]