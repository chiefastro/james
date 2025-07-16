"""
Main FastAPI application entry point for the Conscious Agent System.
"""

from fastapi import FastAPI

app = FastAPI(title="Conscious Agent API", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "Conscious Agent System API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}