# TREC Question Classification — FastAPI Demo

Demo web phân loại câu hỏi 6 nhóm TREC (ABBR / DESC / ENTY / HUM / LOC / NUM) bằng **2 model**:

| Model | Kiến trúc | Test Acc | Self-contained? |
|---|---|---|---|
| Model 1 | CNN-Text (Kim 2014) | **84.8%** | ✅ Có (chỉ cần 1 file .pt) |
| Model 2 | BiLSTM + Multi-Head Attention + ELMo | **94.8%** | ❌ Cần thêm ELMo (~360MB) |

Hỗ trợ input **tiếng Việt** — tự động dịch sang Anh bằng `vinai/vinai-translate-vi2en-v2` rồi mới phân loại.

---

## 🚀 Quick start

### 0. Yêu cầu

- **Python 3.9** (allennlp 2.10 không tương thích Python ≥ 3.10)
- ~3GB RAM (~6GB nếu chạy GPU)
- GPU NVIDIA (optional, CUDA 11.x) — chạy CPU vẫn được, chỉ chậm hơn

### 1. Cài dependencies

**Cách A — venv (khuyên dùng):**

```bash
python3.9 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Torch GPU (CUDA 11.6) — nếu có NVIDIA card:
pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# Hoặc Torch CPU-only:
# pip install torch==1.12.1

# Phần còn lại:
pip install -r requirements.txt
```

**Cách B — conda:**

```bash
conda create -n trec_qc python=3.9 -y
conda activate trec_qc
pip install -r requirements.txt
```

### 2. Đặt model weights vào đúng chỗ

Cấu trúc thư mục **bắt buộc**:

```
project_root/
├── fastapi_app/                   ← code app
│   ├── app.py
│   ├── start_server.py
│   ├── index.html
│   ├── requirements.txt
│   └── models/
│       ├── classifier.py
│       └── translator.py
├── checkpoints/
│   ├── model1_best.pt             ← CNN-Text (~16MB)
│   └── model2_best.pt             ← BiLSTM + ELMo (~48MB)
└── data/
    └── elmo/
        ├── elmo_options.json      ← config nhỏ
        └── elmo_weights.hdf5      ← ~360MB
```

> ⚠️ **`model2_best.pt` KHÔNG đứng một mình được** — nó chỉ chứa BiLSTM layer, **cần thêm ELMo files** để encode text → 1024-d vector.
>
> ELMo sẽ được `start_server.py` tự download nếu chưa có. Hoặc tải tay từ:
> - https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
> - https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5

Model dịch (`vinai-translate-vi2en-v2` ~1.6GB) được HuggingFace cache tự động về `~/.cache/huggingface/` lần đầu khởi động.

### 3. Chạy server

```bash
cd fastapi_app
python start_server.py
```

Server lên ở **http://localhost:8000/**

Lần đầu chạy mất ~1-2 phút (download ELMo + load model). Lần sau ~10-20s.

---

## 📡 API Endpoints

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Trang HTML demo |
| GET | `/health` | Trạng thái load của các model |
| GET | `/docs` | OpenAPI Swagger UI |
| GET | `/api` | Liệt kê endpoints |
| POST | `/predict` | Auto-detect VI/EN, predict |
| POST | `/predict-vi` | Force input là tiếng Việt |
| POST | `/predict-en` | Force input là tiếng Anh |
| POST | `/predict-compare` | Chạy cả 2 model song song |
| POST | `/predict-batch` | Batch nhiều câu hỏi |

### Body của `/predict`

```json
{
  "question": "NASA là viết tắt của gì?",
  "model": "bilstm"   // "cnn" | "bilstm" (default) | "both"
}
```

### Ví dụ với `curl`

```bash
# Predict Model 2 (default)
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"question": "Tháp Eiffel ở đâu?"}'

# Predict Model 1 (CNN)
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"question": "Tháp Eiffel ở đâu?", "model": "cnn"}'

# Compare both
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"question": "Tháp Eiffel ở đâu?", "model": "both"}'
```

---

## 🐳 Docker (tùy chọn — đóng gói đầy đủ)

```bash
docker build -t trec-qc-fastapi .
docker run -p 8000:8000 \
  -v $(pwd)/../checkpoints:/app/checkpoints \
  -v $(pwd)/../data:/app/data \
  trec-qc-fastapi
```

Lưu ý mount checkpoints + data từ ngoài vào để image gọn nhẹ.

---

## 🛠 Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `ImportError: cannot import name 'ModelMetaclass' from 'pydantic.main'` | Pydantic v2 không tương thích allennlp/spacy | `pip install 'pydantic<2'` |
| `OSError: We couldn't connect to 'https://huggingface.co'` | transformers cũ + HF URL mới | Upgrade `pip install 'transformers>=4.30'` |
| `Model 2 (BiLSTM+ELMo) not loaded` ở `/health` | Thiếu ELMo files | Chạy `python start_server.py` để tự download, hoặc kiểm tra `data/elmo/` |
| `Translator not loaded` | Lần đầu load vinai model thất bại (mạng) | Chạy lại lần 2, hoặc download tay model: `huggingface-cli download vinai/vinai-translate-vi2en-v2` |
| Confidence Model 2 rất thấp (<50%) | Bản dịch chứa contraction ("Who's") chưa được expand | Đã fix trong `classifier.py:expand_contractions()` |

---

## 📦 Triển khai cho người khác

Gói 1 archive gồm:
- `fastapi_app/` (toàn bộ folder này)
- `checkpoints/model1_best.pt` (16MB)
- `checkpoints/model2_best.pt` (48MB)
- `data/elmo/elmo_options.json` (1KB) + `data/elmo/elmo_weights.hdf5` (360MB) — *hoặc bỏ qua, để `start_server.py` tự download*

Người nhận chỉ cần:
1. Cài Python 3.9
2. `pip install -r fastapi_app/requirements.txt` (+ torch riêng)
3. `cd fastapi_app && python start_server.py`

Tổng size archive: ~430MB nếu kèm ELMo, ~70MB nếu không.
