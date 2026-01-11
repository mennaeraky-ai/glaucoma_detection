import os
from pathlib import Path

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


# Default model location (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "LAST_glaucoma_model.keras"

# Google Drive file id for optional auto-download
GDRIVE_FILE_ID = "1OAZgc2VA9DBXALdDItvhVt-JFGeNT5JI"


def _cache_resource_if_available(fn):
    if st is not None:
        return st.cache_resource(fn)
    return fn


@_cache_resource_if_available
def get_model_path() -> str:
    """
    Returns a local filesystem path to the trained model.

    Priority:
    1) GLAUCOMA_MODEL_PATH env var (if set)
    2) ./models/LAST_glaucoma_model.keras (repo-relative)

    If missing, attempts a Google Drive download **only** if `gdown` is installed.
    Otherwise, raises FileNotFoundError with a helpful message.
    """
    env_path = os.getenv("GLAUCOMA_MODEL_PATH")
    model_path = Path(env_path).expanduser().resolve() if env_path else _DEFAULT_MODEL_PATH

    if model_path.exists():
        return str(model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Try optional download path (kept non-fatal if gdown is missing)
    try:
        import gdown  # lazy import so app can start without it
    except ModuleNotFoundError:
        raise FileNotFoundError(
            "Model file not found.\n"
            f"- Looked for: {model_path}\n"
            "- Fix: either place the .keras file there, or set env var GLAUCOMA_MODEL_PATH.\n"
            "- Optional: `pip install gdown` to enable auto-download."
        )

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    if st is not None:
        st.info("⬇️ Downloading model from Google Drive...")
    gdown.download(url, str(model_path), quiet=False)

    if not model_path.exists():
        raise FileNotFoundError(
            "Model download did not produce a file.\n"
            f"- Expected at: {model_path}\n"
            "- Fix: download/copy the model manually, or verify the Google Drive file id."
        )

    return str(model_path)
