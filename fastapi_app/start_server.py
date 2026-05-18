"""Quick start script — tự download ELMo weights nếu chưa có rồi khởi động FastAPI.

Chạy:
    cd fastapi_app
    python start_server.py
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELMO_DIR = PROJECT_ROOT / "data" / "elmo"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

ELMO_OPTIONS_URL = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHTS_URL = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


def download_file(url: str, filepath: Path) -> bool:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        print(f"  [OK] {filepath} (đã có)")
        return True
    print(f"  Downloading {url} -> {filepath} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  [OK] Downloaded.")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    print("=" * 60)
    print("  TREC Question Classification — Quick Start")
    print("=" * 60)

    # 1) ELMo
    print("\n[1/3] Kiểm tra ELMo weights...")
    ok1 = download_file(ELMO_OPTIONS_URL, ELMO_DIR / "elmo_options.json")
    ok2 = download_file(ELMO_WEIGHTS_URL, ELMO_DIR / "elmo_weights.hdf5")
    if not (ok1 and ok2):
        print("  [WARN] Không tải được ELMo, server vẫn start nhưng /predict sẽ lỗi.")

    # 2) Checkpoint
    print("\n[2/3] Kiểm tra classifier checkpoint...")
    ck_best = CHECKPOINT_DIR / "model2_best.pt"
    ck_last = CHECKPOINT_DIR / "model2_last.pt"
    if ck_best.exists():
        print(f"  [OK] {ck_best}")
    elif ck_last.exists():
        print(f"  [OK] {ck_last}")
    else:
        print(f"  [WARN] Không tìm thấy {ck_best} hoặc {ck_last}")
        print("  Hãy copy file checkpoint vào thư mục checkpoints/")

    # 3) Dependencies
    print("\n[3/3] Kiểm tra dependencies...")
    required = ["fastapi", "uvicorn", "torch", "transformers", "allennlp", "h5py"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  [MISSING] {pkg}")
    if missing:
        print(f"\n  Cài thiếu: pip install {' '.join(missing)}")
        print("  (Lưu ý: allennlp 2.10.1 cần torch < 1.13)")

    # Start server
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 60)
    print(f"  Starting FastAPI Server @ http://0.0.0.0:{port}")
    print("=" * 60)
    print(f"\n  Demo UI : http://localhost:{port}/")
    print(f"  Docs    : http://localhost:{port}/docs")
    print(f"  Health  : http://localhost:{port}/health")
    print()

    os.chdir(Path(__file__).parent)  # để app.py mở được index.html
    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
    except Exception as e:
        print(f"[ERROR] {e}")
        print(f"Thử: python -m uvicorn app:app --host 0.0.0.0 --port {port}")


if __name__ == "__main__":
    main()
