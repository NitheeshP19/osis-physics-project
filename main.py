from pathlib import Path

from app.main import app
from app.ml.model_loader import registry


registry.load_models(str(Path(__file__).resolve().parent))
