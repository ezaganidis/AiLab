from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "saved_models"
PIPELINES_DIR = ARTIFACTS_DIR / "saved_pipelines"
IMAGES_DIR = ROOT_DIR / "images"
LOGO_PATH = IMAGES_DIR / "logo.svg"


def ensure_app_dirs() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
