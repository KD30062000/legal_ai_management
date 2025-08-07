from fastapi import FastAPI
from app.routes.upload import router as upload_router

app = FastAPI(title="Legal Document AI")

app.include_router(upload_router, prefix="/api")
