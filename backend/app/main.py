# from fastapi import FastAPI
# from app.routes.upload import router as upload_router

# app = FastAPI(title="Legal Document AI")

# app.include_router(upload_router, prefix="/api")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.upload import router as upload_router
from app.routes.chat import router as chat_router
import os
from app.models.database import create_tables

# Create tables on startup
create_tables()

app = FastAPI(title="Legal Document AI", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(chat_router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Legal Document AI API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}