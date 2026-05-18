# =====================================================================
# TREC Question Classification — FastAPI Dockerfile
#
# Build:
#   cd fastapi_app
#   docker build -t trec-qc .
#
# Run (mount checkpoints + ELMo từ host):
#   docker run -p 8000:8000 \
#       -v $(pwd)/../checkpoints:/project/checkpoints \
#       -v $(pwd)/../data:/project/data \
#       trec-qc
#
# Truy cập: http://localhost:8000
# =====================================================================

FROM python:3.9-slim

# Cài system deps cần cho allennlp/h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /project/fastapi_app

# Cài torch CPU trước (cache layer riêng)
RUN pip install --no-cache-dir torch==1.12.1

# Cài các deps còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code (checkpoints + ELMo nên mount từ host, không build vào image)
COPY app.py start_server.py index.html ./
COPY models/ ./models/

# Pre-download translator model để startup nhanh (tùy chọn, +1.6GB image)
# RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
#     AutoTokenizer.from_pretrained('vinai/vinai-translate-vi2en-v2', src_lang='vi_VN'); \
#     AutoModelForSeq2SeqLM.from_pretrained('vinai/vinai-translate-vi2en-v2')"

EXPOSE 8000

CMD ["python", "start_server.py"]
