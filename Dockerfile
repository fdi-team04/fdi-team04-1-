FROM python:3.14.6
# Colab과 버전 맞추기

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/hf

# CPU 전용 torch 먼저 → requirements의 torch 줄은 이미 설치된 걸로 처리됨
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# 코드는 COPY하지 않음 (compose에서 연결)