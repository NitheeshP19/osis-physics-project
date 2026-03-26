# app/main.py
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.ml.model_loader import registry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(title="OSIS Hybrid Simulation Platform", version="2.0")

# Security Headers & CORS for Render Deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Hook (Preload ML Models Globally)
@app.on_event("startup")
def startup_event():
    logger.info("Starting up FastAPI service...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # points to c:\Users\dell\Documents\physics
    registry.load_models(base_dir)

# Static file routing
static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "OSIS Simulation API is running."}

# Include all grouped simulation routes
app.include_router(router)
