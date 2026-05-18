"""
Streamlit demo cho TREC Question Classification.

Chạy:
    /home/xuand/miniconda3/envs/NLP/bin/streamlit run app.py
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
ELMO_DIR = ROOT / "data" / "elmo"

LABEL_NAMES = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
LABEL_DESCRIPTIONS = {
    "ABBR": "Abbreviation — viết tắt (e.g. What does NASA stand for?)",
    "DESC": "Description — định nghĩa, giải thích (e.g. What is photosynthesis?)",
    "ENTY": "Entity — sự vật, sự kiện (e.g. What films featured Popeye Doyle?)",
    "HUM":  "Human — người, nhóm người (e.g. Who was Galileo?)",
    "LOC":  "Location — địa điểm (e.g. Where is the Eiffel Tower?)",
    "NUM":  "Numeric — số, ngày tháng (e.g. How far is Denver from Aspen?)",
}

st.set_page_config(
    page_title="TREC Question Classification Demo",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
#                     MODEL DEFINITIONS
# ============================================================
class CNN_Text(nn.Module):
    """Model 1: CNN-text (Kim 2014)."""

    def __init__(self, vocab_size, embedding_dim=128, num_filters=128,
                 kernel_sizes=(3, 4, 5), num_classes=6, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        outs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            c = F.max_pool1d(c, c.shape[2]).squeeze(2)
            outs.append(c)
        return self.fc(self.dropout(torch.cat(outs, dim=1)))


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, lstm_out):
        q = self.query.expand(lstm_out.size(0), -1, -1)
        ctx, w = self.attn(q, lstm_out, lstm_out)
        return ctx.squeeze(1), w.squeeze(1)


class BiLSTMAttentionELMo(nn.Module):
    """Model 2: BiLSTM + Multi-Head Attention + ELMo."""

    def __init__(self, elmo_dim=1024, hidden_dim=256, num_classes=6,
                 num_layers=2, num_heads=4, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(elmo_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention = MultiHeadAttentionPool(hidden_dim * 2, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.dropout(F.relu(self.input_proj(x)))
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        ctx, w = self.attention(lstm_out)
        return self.fc(self.dropout(ctx)), w


# ============================================================
#                          LOADERS
# ============================================================
CONTRACTIONS = {
    "who's": "who is", "what's": "what is", "where's": "where is",
    "when's": "when is", "how's": "how is", "why's": "why is",
    "that's": "that is", "it's": "it is", "there's": "there is",
    "he's": "he is", "she's": "she is", "let's": "let us",
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
}


def expand_contractions(text: str) -> str:
    """TREC training chỉ có dạng đầy đủ ('Who was', 'What is') —
    expand contractions để match distribution train của model."""
    out = text
    for k, v in CONTRACTIONS.items():
        out = re.sub(re.escape(k), v, out, flags=re.IGNORECASE)
    return out


def preprocess_text(text: str) -> str:
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Đang load Model 1 (CNN)...")
def load_model1():
    ckpt = torch.load(CKPT_DIR / "model1_best.pt", map_location="cpu")
    vocab = ckpt["vocab"]
    sd = ckpt["model_state_dict"]
    vocab_size, emb_dim = sd["embedding.weight"].shape
    model = CNN_Text(vocab_size=vocab_size, embedding_dim=emb_dim,
                     num_filters=128, kernel_sizes=(3, 4, 5),
                     num_classes=6, dropout=0.5)
    model.load_state_dict(sd)
    model.eval()
    return model, vocab, {"best_val_acc": ckpt.get("val_acc"),
                          "epoch": ckpt.get("epoch")}


@st.cache_resource(show_spinner="Đang load ELMo + Model 2 (BiLSTM)...")
def load_model2():
    from allennlp.modules.elmo import Elmo, batch_to_ids  # noqa: F401
    ckpt = torch.load(CKPT_DIR / "model2_best.pt", map_location="cpu")
    cfg = ckpt["config"]
    model = BiLSTMAttentionELMo(
        elmo_dim=cfg["ELMO_DIM"], hidden_dim=cfg["HIDDEN_DIM"],
        num_classes=cfg["NUM_CLASSES"], num_layers=cfg["NUM_LAYERS"],
        num_heads=cfg["NUM_HEADS"], dropout=cfg["DROPOUT"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    elmo = Elmo(
        options_file=str(ELMO_DIR / "elmo_options.json"),
        weight_file=str(ELMO_DIR / "elmo_weights.hdf5"),
        num_output_representations=1, dropout=0.0,
    )
    elmo.eval()
    return model, elmo, batch_to_ids, cfg, {
        "best_val_acc": ckpt.get("best_val_acc"),
        "epoch": ckpt.get("epoch"),
        "label_names": ckpt.get("label_names", LABEL_NAMES),
    }


@st.cache_resource(show_spinner="Đang load mô hình dịch VI→EN (vinai-translate-vi2en-v2)...")
def load_translator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
    mdl.eval()
    en_id = tok.lang_code_to_id["en_XX"] if hasattr(tok, "lang_code_to_id") \
        else tok.convert_tokens_to_ids("en_XX")
    return tok, mdl, en_id


@st.cache_data(show_spinner=False)
def translate_vi_to_en(vi_text: str) -> str:
    tok, mdl, en_id = load_translator()
    inputs = tok(vi_text, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = mdl.generate(
            **inputs, decoder_start_token_id=en_id,
            num_beams=5, max_length=64, early_stopping=True,
        )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


@st.cache_data
def load_results():
    out = {}
    for k, fname in [("m1", "model1_results.json"), ("m2", "model2_results.json")]:
        p = RESULTS_DIR / fname
        if p.exists():
            out[k] = json.loads(p.read_text())
    return out


# ============================================================
#                         INFERENCE
# ============================================================
MAX_LEN = 64


def predict_model1(model, vocab, raw_text: str):
    text = preprocess_text(raw_text)
    tokens = text.split()
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
    padded = tokens + ["<pad>"] * (MAX_LEN - len(tokens))
    unk = vocab.get("<unk>", 1)
    ids = [vocab.get(t, unk) for t in padded]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    return tokens, probs


def predict_model2(model, elmo, batch_to_ids, raw_text: str):
    raw_text = expand_contractions(raw_text)
    tokens = raw_text.split()
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
    padded = tokens + ["<pad>"] * (MAX_LEN - len(tokens))
    char_ids = batch_to_ids([padded])
    with torch.no_grad():
        emb = elmo(char_ids)["elmo_representations"][0]
        logits, attn_weights = model(emb)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
        attn = attn_weights.squeeze(0).numpy()
    return tokens, probs, attn[: len(tokens)]


# ============================================================
#                          UI
# ============================================================
st.title("🎯 TREC Question Classification — Demo")
st.caption(
    "Phân loại câu hỏi tiếng Anh vào 6 nhóm: ABBR / DESC / ENTY / HUM / LOC / NUM. "
    "Bằng 2 mô hình: **CNN-Text (Kim 2014)** và **BiLSTM + Multi-Head Attention + ELMo**."
)

# ---- Sidebar ----
with st.sidebar:
    st.header("Cấu hình")
    model_choice = st.radio(
        "Chọn mô hình",
        ["Model 1: CNN-Text (baseline)",
         "Model 2: BiLSTM + Attention + ELMo (SOTA)",
         "So sánh cả hai"],
        index=2,
    )
    st.divider()
    st.subheader("Nhãn TREC")
    for k, v in LABEL_DESCRIPTIONS.items():
        st.markdown(f"**{k}** — {v[len(k) + 3:]}")

results = load_results()

# ---- Metrics block ----
with st.expander("📊 Kết quả test (đã train sẵn)", expanded=False):
    c1, c2 = st.columns(2)
    if "m1" in results:
        r = results["m1"]
        with c1:
            st.markdown("### Model 1 — CNN-Text")
            st.metric("Test Accuracy", f"{r['test_accuracy']*100:.2f}%")
            st.metric("F1 (macro)", f"{r['f1_macro']:.4f}")
            st.metric("F1 (weighted)", f"{r['f1_weighted']:.4f}")
            st.caption(f"Params: {r['total_params']:,} | Vocab: {r['vocab_size']}")
    if "m2" in results:
        r = results["m2"]
        with c2:
            st.markdown("### Model 2 — BiLSTM + ELMo")
            st.metric("Test Accuracy", f"{r['test_accuracy']*100:.2f}%")
            st.metric("F1 (macro)", f"{r['f1_macro']:.4f}")
            st.metric("F1 (weighted)", f"{r['f1_weighted']:.4f}")
            st.caption(f"Params: {r['total_params']:,} | Embedding: {r['embedding']}")

# ---- Input ----
st.divider()
st.subheader("Nhập câu hỏi")

vi_mode = st.toggle(
    "🇻🇳 Nhập câu hỏi bằng **tiếng Việt** (tự dịch sang Anh bằng `vinai-translate-vi2en-v2`)",
    value=False,
)

if vi_mode:
    examples = [
        "NASA là viết tắt của gì ?",
        "Quang hợp là gì ?",
        "Ai là tổng thống đầu tiên của nước Mỹ ?",
        "Tháp Eiffel ở đâu ?",
        "Khoảng cách từ Hà Nội đến Đà Nẵng là bao nhiêu ?",
        "Bộ phim nào có nhân vật Popeye Doyle ?",
    ]
    placeholder = "NASA là viết tắt của gì ?"
    label = "Câu hỏi (Tiếng Việt):"
else:
    examples = [
        "What does NASA stand for ?",
        "What is photosynthesis ?",
        "Who was the first president of the United States ?",
        "Where is the Eiffel Tower located ?",
        "How far is it from Denver to Aspen ?",
        "What films featured the character Popeye Doyle ?",
    ]
    placeholder = "What does NASA stand for ?"
    label = "Câu hỏi (English):"

col_input, col_example = st.columns([3, 1])
with col_example:
    chosen = st.selectbox("Ví dụ mẫu", ["(tự nhập)"] + examples)
default_text = "" if chosen == "(tự nhập)" else chosen
with col_input:
    question = st.text_input(label, value=default_text, placeholder=placeholder)

run = st.button("Phân loại", type="primary", use_container_width=True)


def render_prediction(name, tokens, probs, attn=None):
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    conf = probs[pred_idx] * 100

    st.markdown(f"### {name}")
    st.success(f"**Nhãn dự đoán: `{pred_label}`** — {LABEL_DESCRIPTIONS[pred_label]}  \nĐộ tin cậy: **{conf:.2f}%**")

    df = pd.DataFrame({"Label": LABEL_NAMES, "Probability": probs})
    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    st.bar_chart(df.set_index("Label"))

    with st.expander("Xác suất chi tiết"):
        st.dataframe(
            df.assign(Probability=lambda d: (d["Probability"] * 100).round(2).astype(str) + " %"),
            hide_index=True, use_container_width=True,
        )

    if attn is not None and len(tokens) > 0:
        st.markdown("**Attention weights** (Model 2):")
        attn_df = pd.DataFrame({"token": tokens, "weight": attn})
        # Normalize for color
        max_w = float(attn.max()) if attn.max() > 0 else 1.0
        html = "<div style='font-size:18px; line-height:2.2;'>"
        for tok, w in zip(tokens, attn):
            alpha = float(w) / max_w
            html += (
                f"<span style='background-color: rgba(255, 165, 0, {alpha:.2f}); "
                f"padding: 3px 6px; margin: 2px; border-radius: 4px;' "
                f"title='{w:.4f}'>{tok}</span> "
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        st.bar_chart(attn_df.set_index("token"))


if run:
    if not question.strip():
        st.warning("Hãy nhập câu hỏi.")
        st.stop()

    if vi_mode:
        with st.spinner("Đang dịch tiếng Việt → tiếng Anh..."):
            en_question = translate_vi_to_en(question.strip())
        st.info(
            f"🇻🇳 **VI**: {question.strip()}  \n"
            f"🇬🇧 **EN** (dịch tự động): {en_question}"
        )
        question_for_model = en_question
    else:
        question_for_model = question

    use_m1 = model_choice.startswith("Model 1") or model_choice.startswith("So sánh")
    use_m2 = model_choice.startswith("Model 2") or model_choice.startswith("So sánh")

    cols = st.columns(2) if model_choice.startswith("So sánh") else [st]

    if use_m1:
        try:
            m1, vocab, m1_meta = load_model1()
            toks1, probs1 = predict_model1(m1, vocab, question_for_model)
            with cols[0]:
                render_prediction("Model 1 — CNN-Text", toks1, probs1)
        except Exception as e:
            st.error(f"Lỗi Model 1: {e}")

    if use_m2:
        try:
            m2, elmo, batch_to_ids, cfg, m2_meta = load_model2()
            toks2, probs2, attn = predict_model2(m2, elmo, batch_to_ids, question_for_model)
            col = cols[1] if model_choice.startswith("So sánh") else cols[0]
            with col:
                render_prediction("Model 2 — BiLSTM + ELMo", toks2, probs2, attn=attn)
        except FileNotFoundError as e:
            st.error(f"Không tìm thấy file ELMo: {e}")
        except Exception as e:
            st.error(f"Lỗi Model 2: {e}")

st.divider()
st.caption(
    "TREC dataset · 6 coarse labels · Model 1: CNN-Text (Kim 2014, CNN-rand 128d) · "
    "Model 2: BiLSTM 2-layer + Multi-Head Attention 4 heads + ELMo 1024d contextual."
)
