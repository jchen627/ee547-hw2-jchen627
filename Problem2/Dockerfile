FROM python:3.11-slim

# 安装依赖
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app
COPY train_embeddings.py /app/
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import torch; print(torch.__version__)"

ENTRYPOINT ["python", "/app/train_embeddings.py"]

