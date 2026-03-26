# app/ml/model_loader.py
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Singleton registry to ensure ML models are loaded exactly once in memory for Gunicorn/Uvicorn."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.model_lower = None
            cls._instance.model_upper = None
            cls._instance.feature_columns = None
            cls._instance.explainer = None
            cls._instance.is_loaded = False
        return cls._instance

    def load_models(self, base_dir: str):
        if self.is_loaded:
            return
            
        try:
            logger.info("Loading ML Models globally into memory...")
            self.model = joblib.load(os.path.join(base_dir, "osis_snr_model.pkl"))
            self.model_lower = joblib.load(os.path.join(base_dir, "osis_snr_model_lower.pkl"))
            self.model_upper = joblib.load(os.path.join(base_dir, "osis_snr_model_upper.pkl"))
            self.feature_columns = joblib.load(os.path.join(base_dir, "osis_features.pkl"))
            self.explainer = joblib.load(os.path.join(base_dir, "osis_explainer.pkl"))
            self.is_loaded = True
            logger.info("All ML models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise

registry = ModelRegistry()
