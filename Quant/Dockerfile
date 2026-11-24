FROM python:3.11-slim
# Colab과 버전 맞추기

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

# 코드는 COPY하지 않음 (compose에서 연결)
