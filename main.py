from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.ml.model_loader import registry
from app.api.routes import router as api_router

# Load registry models globally so prediction functions in app/ml/predictor.py work
registry.load_models(".")

app = FastAPI(title="OSIS Hybrid SNR Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5501", "http://localhost:5501", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

app.include_router(api_router)
