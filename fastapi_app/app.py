"""FastAPI server cho TREC Question Classification.

- /predict       : auto-detect VI/EN
- /predict-vi    : input tiếng Việt (dịch sang Anh rồi phân loại)
- /predict-en    : input tiếng Anh (phân loại trực tiếp)
- /predict-batch : batch
- /health        : trạng thái load model
- /              : trang HTML demo
"""

import os
import time
import unicodedata
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.translator import get_translator
from models.classifier import get_classifier, get_cnn_classifier


app = FastAPI(
    title="TREC Question Classification API",
    description="Phân loại câu hỏi 6 nhóm TREC (ABBR/DESC/ENTY/HUM/LOC/NUM). "
                "Câu hỏi tiếng Việt sẽ được tự động dịch sang tiếng Anh "
                "bằng vinai-translate-vi2en-v2 trước khi phân loại.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

translator = None
classifier = None       # Model 2: BiLSTM + Attention + ELMo
classifier_cnn = None   # Model 1: CNN-Text


class PredictionRequest(BaseModel):
    question: str = Field(..., example="NASA là viết tắt của gì?")
    model: Optional[str] = Field(
        default="bilstm",
        description="Chọn model: 'cnn' (Model 1) | 'bilstm' (Model 2, default) | 'both'",
    )


class PredictionResponse(BaseModel):
    success: bool = True
    question: str
    language: str
    translation: Optional[str] = None
    model: str
    predicted_class: str
    confidence: float
    all_probabilities: dict
    processing_time_ms: int


class ComparisonResponse(BaseModel):
    success: bool = True
    question: str
    language: str
    translation: Optional[str] = None
    cnn: dict
    bilstm: dict
    processing_time_ms: int


# ============================================================
def detect_vietnamese(text: str) -> bool:
    """Detect tiếng Việt qua các dấu thanh/dấu mũ trong NFC string."""
    if not text:
        return False
    nfd = unicodedata.normalize("NFD", text)
    diacritic_count = sum(1 for c in nfd if unicodedata.category(c) == "Mn")
    has_d_stroke = "đ" in text.lower() or "Đ" in text
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count == 0:
        return False
    return has_d_stroke or (diacritic_count / max(alpha_count, 1)) > 0.05


# ============================================================
@app.on_event("startup")
async def startup_event():
    global translator, classifier, classifier_cnn
    print("=" * 60)
    print("LOADING MODELS...")
    print("=" * 60)

    try:
        translator = get_translator()
        print("[OK] Translator loaded (vinai-translate-vi2en-v2)")
    except Exception as e:
        print(f"[WARN] Translator not loaded: {e}")
        translator = None

    try:
        classifier_cnn = get_cnn_classifier()
        print("[OK] Model 1 loaded (CNN-Text)")
    except Exception as e:
        print(f"[WARN] Model 1 not loaded: {e}")
        classifier_cnn = None

    try:
        ckpt_path = os.environ.get("CHECKPOINT_PATH")
        classifier = get_classifier(ckpt_path)
        print("[OK] Model 2 loaded (BiLSTM + Attention + ELMo)")
    except Exception as e:
        print(f"[WARN] Model 2 not loaded: {e}")
        classifier = None
    print("=" * 60)


# ============================================================
@app.get("/health")
async def health_check():
    import torch
    return {
        "status": "healthy",
        "models": {
            "translator": "loaded" if translator else "not loaded",
            "cnn (Model 1)": "loaded" if classifier_cnn else "not loaded",
            "bilstm (Model 2)": "loaded" if classifier else "not loaded",
        },
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
    }


# ============================================================
def _pick_classifier(model_name: str):
    name = (model_name or "bilstm").lower()
    if name == "cnn":
        if classifier_cnn is None:
            raise HTTPException(status_code=503, detail="Model 1 (CNN) not loaded.")
        return classifier_cnn, "cnn"
    if name in ("bilstm", "elmo"):
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model 2 (BiLSTM+ELMo) not loaded.")
        return classifier, "bilstm"
    raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'. Use 'cnn' or 'bilstm'.")


def _translate_if_needed(question: str, force_lang: Optional[str] = None):
    """Return (text_for_model, language, translation_or_None)."""
    lang = force_lang or ("vi" if detect_vietnamese(question) else "en")
    if lang == "vi":
        if translator is None:
            raise HTTPException(status_code=503,
                                detail="Translator not loaded (vinai model).")
        translation = translator.translate(question.strip(), target_lang="en")
        return translation, lang, translation
    return question, lang, None


def _do_predict(question: str, model_name: str = "bilstm",
                force_lang: Optional[str] = None) -> PredictionResponse:
    start = time.time()
    clf, used_model = _pick_classifier(model_name)
    text_for_model, lang, translation = _translate_if_needed(question, force_lang)
    result = clf.predict(text_for_model)
    return PredictionResponse(
        question=question, language=lang, translation=translation,
        model=used_model,
        predicted_class=result["class"],
        confidence=result["confidence"],
        all_probabilities=result["all_probs"],
        processing_time_ms=int((time.time() - start) * 1000),
    )


def _do_compare(question: str, force_lang: Optional[str] = None) -> ComparisonResponse:
    if classifier is None or classifier_cnn is None:
        raise HTTPException(status_code=503,
                            detail="Cần cả 2 model để so sánh — kiểm tra /health.")
    start = time.time()
    text_for_model, lang, translation = _translate_if_needed(question, force_lang)
    r_cnn = classifier_cnn.predict(text_for_model)
    r_bilstm = classifier.predict(text_for_model)
    return ComparisonResponse(
        question=question, language=lang, translation=translation,
        cnn={
            "predicted_class": r_cnn["class"],
            "confidence": r_cnn["confidence"],
            "all_probabilities": r_cnn["all_probs"],
        },
        bilstm={
            "predicted_class": r_bilstm["class"],
            "confidence": r_bilstm["confidence"],
            "all_probabilities": r_bilstm["all_probs"],
        },
        processing_time_ms=int((time.time() - start) * 1000),
    )


@app.post("/predict")
async def predict(request: PredictionRequest):
    if (request.model or "").lower() == "both":
        return _do_compare(request.question)
    return _do_predict(request.question, model_name=request.model or "bilstm")


@app.post("/predict-vi")
async def predict_vi(request: PredictionRequest):
    if (request.model or "").lower() == "both":
        return _do_compare(request.question, force_lang="vi")
    return _do_predict(request.question, model_name=request.model or "bilstm",
                       force_lang="vi")


@app.post("/predict-en")
async def predict_en(request: PredictionRequest):
    if (request.model or "").lower() == "both":
        return _do_compare(request.question, force_lang="en")
    return _do_predict(request.question, model_name=request.model or "bilstm",
                       force_lang="en")


@app.post("/predict-compare", response_model=ComparisonResponse)
async def predict_compare(request: PredictionRequest):
    """Chạy cả Model 1 (CNN) và Model 2 (BiLSTM+ELMo) song song."""
    return _do_compare(request.question)


@app.post("/predict-batch")
async def predict_batch(requests: List[PredictionRequest]):
    start = time.time()
    results = []
    for req in requests:
        try:
            r = _do_predict(req.question, model_name=req.model or "bilstm")
            results.append(r.dict())
        except HTTPException as e:
            results.append({"success": False, "question": req.question,
                            "error": e.detail})
    return {
        "total": len(results),
        "processing_time_ms": int((time.time() - start) * 1000),
        "results": results,
    }


# ============================================================
@app.get("/api")
async def api_info():
    return {
        "message": "TREC Question Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Trạng thái load model",
            "/predict": "Auto detect VI/EN rồi phân loại",
            "/predict-vi": "Force tiếng Việt (dịch trước)",
            "/predict-en": "Force tiếng Anh",
            "/predict-batch": "Batch prediction",
        },
    }


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
