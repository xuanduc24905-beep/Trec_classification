"""TREC question classifier — BiLSTM + Multi-Head Attention + ELMo (Model 2).

Pipeline khi inference:
    text -> expand_contractions -> tokenize+pad
         -> ELMo (allennlp) -> 1024-d vector
         -> BiLSTM + Attention -> 6 logits -> softmax
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL_NAMES = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
MAX_LEN = 64

# TREC train chỉ có câu hỏi dạng đầy đủ ("Who is", "What was") —
# expand contractions để model thấy distribution quen thuộc.
_CONTRACTIONS = {
    "who's": "who is", "what's": "what is", "where's": "where is",
    "when's": "when is", "how's": "how is", "why's": "why is",
    "that's": "that is", "it's": "it is", "there's": "there is",
    "he's": "he is", "she's": "she is", "let's": "let us",
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
}


def expand_contractions(text: str) -> str:
    out = text
    for k, v in _CONTRACTIONS.items():
        out = re.sub(re.escape(k), v, out, flags=re.IGNORECASE)
    return out


def _preprocess_for_cnn(text: str) -> str:
    """Lowercase + bỏ ký tự ngoài a-z0-9 (giống pipeline train Model 1)."""
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class _CNN_Text(nn.Module):
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


class CNNClassifier:
    """Model 1 — CNN-Text. Self-contained: text -> word ids -> CNN -> logits."""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.vocab = ckpt["vocab"]
        sd = ckpt["model_state_dict"]
        vocab_size, emb_dim = sd["embedding.weight"].shape
        self.model = _CNN_Text(
            vocab_size=vocab_size, embedding_dim=emb_dim,
            num_filters=128, kernel_sizes=(3, 4, 5),
            num_classes=6, dropout=0.5,
        ).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

    def predict(self, text: str) -> dict:
        text = _preprocess_for_cnn(text)
        tokens = text.split()[:MAX_LEN]
        padded = tokens + ["<pad>"] * (MAX_LEN - len(tokens))
        unk = self.vocab.get("<unk>", 1)
        ids = [self.vocab.get(t, unk) for t in padded]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(probs.argmax())
        return {
            "class": LABEL_NAMES[idx],
            "confidence": float(probs[idx]),
            "all_probs": {LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)},
            "tokens": tokens,
        }


class _MultiHeadAttentionPool(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, lstm_out):
        q = self.query.expand(lstm_out.size(0), -1, -1)
        ctx, w = self.attn(q, lstm_out, lstm_out)
        return ctx.squeeze(1), w.squeeze(1)


class _BiLSTMAttentionELMo(nn.Module):
    def __init__(self, elmo_dim=1024, hidden_dim=256, num_classes=6,
                 num_layers=2, num_heads=4, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(elmo_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention = _MultiHeadAttentionPool(hidden_dim * 2, num_heads)
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


class TextClassifier:
    def __init__(self, checkpoint_path: str,
                 elmo_options: str, elmo_weights: str,
                 device: str = None):
        from allennlp.modules.elmo import Elmo, batch_to_ids
        self._batch_to_ids = batch_to_ids
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt["config"]
        self.label_names = ckpt.get("label_names", LABEL_NAMES)

        self.model = _BiLSTMAttentionELMo(
            elmo_dim=cfg["ELMO_DIM"], hidden_dim=cfg["HIDDEN_DIM"],
            num_classes=cfg["NUM_CLASSES"], num_layers=cfg["NUM_LAYERS"],
            num_heads=cfg["NUM_HEADS"], dropout=cfg["DROPOUT"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.elmo = Elmo(
            options_file=elmo_options, weight_file=elmo_weights,
            num_output_representations=1, dropout=0.0,
        ).to(self.device)
        self.elmo.eval()

    def predict(self, text: str) -> dict:
        text = expand_contractions(text.strip())
        tokens = text.split()[:MAX_LEN]
        padded = tokens + ["<pad>"] * (MAX_LEN - len(tokens))
        char_ids = self._batch_to_ids([padded]).to(self.device)
        with torch.no_grad():
            emb = self.elmo(char_ids)["elmo_representations"][0]
            logits, attn = self.model(emb)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(probs.argmax())
        return {
            "class": self.label_names[idx],
            "confidence": float(probs[idx]),
            "all_probs": {self.label_names[i]: float(p) for i, p in enumerate(probs)},
            "tokens": tokens,
            "attention": attn.squeeze(0).cpu().numpy()[: len(tokens)].tolist(),
        }


_singleton_bilstm: Optional[TextClassifier] = None
_singleton_cnn: Optional[CNNClassifier] = None


def get_cnn_classifier(checkpoint_path: Optional[str] = None) -> CNNClassifier:
    """Load Model 1 (CNN-Text) — self-contained, không cần ELMo."""
    global _singleton_cnn
    if _singleton_cnn is not None:
        return _singleton_cnn

    project_root = Path(__file__).resolve().parents[2]
    ckpt = Path(checkpoint_path) if checkpoint_path else project_root / "checkpoints" / "model1_best.pt"
    if not ckpt.is_absolute():
        ckpt = project_root / ckpt
    if not ckpt.exists():
        alt = project_root / "checkpoints" / "model1_last.pt"
        if alt.exists():
            ckpt = alt
        else:
            raise FileNotFoundError(f"Checkpoint không tồn tại: {ckpt}")
    _singleton_cnn = CNNClassifier(str(ckpt))
    return _singleton_cnn


def get_classifier(checkpoint_path: Optional[str] = None) -> TextClassifier:
    """Load classifier (singleton).

    Paths resolve relative to the project root (parent of fastapi_app/).
    """
    global _singleton_bilstm
    if _singleton_bilstm is not None:
        return _singleton_bilstm

    project_root = Path(__file__).resolve().parents[2]
    ckpt = Path(checkpoint_path) if checkpoint_path else project_root / "checkpoints" / "model2_best.pt"
    if not ckpt.is_absolute():
        ckpt = project_root / ckpt
    if not ckpt.exists():
        alt = project_root / "checkpoints" / "model2_last.pt"
        if alt.exists():
            ckpt = alt
        else:
            raise FileNotFoundError(f"Checkpoint không tồn tại: {ckpt}")

    elmo_opt = project_root / "data" / "elmo" / "elmo_options.json"
    elmo_w = project_root / "data" / "elmo" / "elmo_weights.hdf5"
    if not elmo_opt.exists() or not elmo_w.exists():
        raise FileNotFoundError(
            f"Thiếu ELMo files: {elmo_opt} / {elmo_w}. "
            "Chạy start_server.py để tự download."
        )

    _singleton_bilstm = TextClassifier(str(ckpt), str(elmo_opt), str(elmo_w))
    return _singleton_bilstm
