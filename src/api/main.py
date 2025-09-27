"""
FastAPI Service - PDF Translation Pipeline API
High-precision PDF translation with layout preservation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import asyncio
from datetime import datetime
import os
from pathlib import Path

# Import core components
from ..core import JobManager, TranslationRequest, JobResult
from ..core.schemas.job import JobStatus

# Initialize FastAPI app
app = FastAPI(
    title="PDF Translation Pipeline API",
    description="High-precision PDF translation with layout preservation",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Settings:
    UPLOAD_DIR = Path("data/uploads")
    OUTPUT_DIR = Path("data/outputs")
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS = {".pdf"}

settings = Settings()

# Create directories
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize services
job_manager = JobManager()

# Pydantic models
class TranslationRequestModel(BaseModel):
    """Translation job request"""
    source_lang: str = Field(default="auto", description="Source language code or 'auto'")
    target_lang: str = Field(..., description="Target language code")
    document_type: Optional[str] = Field(default="general", description="Document type")
    preserve_formatting: bool = Field(default=True, description="Preserve formatting")
    use_vla: Optional[bool] = Field(default=None, description="Force VLA usage")
    priority: int = Field(default=5, description="Priority (1-10)")
    callback_url: Optional[str] = Field(default=None, description="Webhook for completion")

class JobStatusModel(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: int
    message: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    result_url: Optional[str]
    error: Optional[str]
    metrics: Optional[Dict[str, Any]]

class TranslationResponse(BaseModel):
    """Translation job creation response"""
    job_id: str
    status: str
    message: str
    estimated_time: int  # seconds

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, bool]
    timestamp: datetime

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "PDF Translation Pipeline",
        "version": "2.0.0",
        "status": "operational"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""

    # Check service health
    services = {
        "api": True,
        "job_manager": True,
        "file_system": settings.UPLOAD_DIR.exists() and settings.OUTPUT_DIR.exists()
    }

    return HealthCheck(
        status="healthy" if all(services.values()) else "degraded",
        version="2.0.0",
        services=services,
        timestamp=datetime.utcnow()
    )

@app.post("/translate", response_model=TranslationResponse)
async def create_translation_job(
    file: UploadFile = File(...),
    request: TranslationRequestModel = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Create a new PDF translation job.

    - **file**: PDF file to translate (max 500MB)
    - **source_lang**: Source language code or 'auto' for detection
    - **target_lang**: Target language code (required)
    - **document_type**: Type of document (general, scientific, legal, technical)
    - **preserve_formatting**: Whether to preserve exact formatting
    - **use_vla**: Force Vision-Language model usage
    - **priority**: Job priority (1-10, higher = faster)
    - **callback_url**: Webhook URL for completion notification
    """

    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Check file size
    file_size = 0
    contents = await file.read()
    file_size = len(contents)

    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Create translation request
    translation_request = TranslationRequest(
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        preserve_formatting=request.preserve_formatting,
        preserve_formulas=True,  # Default to preserve formulas
        preserve_tables=True     # Default to preserve tables
    )

    # Submit job
    try:
        job_id = await job_manager.submit_job(
            filename=file.filename,
            file_data=contents,
            mime_type=file.content_type or "application/pdf",
            request=translation_request
        )

        # Estimate processing time
        estimated_time = estimate_processing_time(
            file_size,
            request.document_type,
            request.use_vla
        )

        return TranslationResponse(
            job_id=job_id,
            status="queued",
            message="Translation job created successfully",
            estimated_time=estimated_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create translation job: {str(e)}"
        )

@app.get("/jobs/{job_id}", response_model=JobStatusModel)
async def get_job_status(job_id: str):
    """
    Get translation job status.

    - **job_id**: Job identifier from translation request
    """

    try:
        job_result = await job_manager.get_job_status(job_id)

        if not job_result:
            raise HTTPException(status_code=404, detail="Job not found")

        # Build response
        response = JobStatusModel(
            job_id=job_result.job_id,
            status=job_result.status.value,
            progress=100 if job_result.status.value == "completed" else 50,  # Simplified
            message=get_status_message(job_result.status.value, 50),
            created_at=job_result.created_at,
            updated_at=job_result.completed_at,
            result_url=None,
            error=job_result.error_message,
            metrics=job_result.metrics
        )

        # Add result URL if completed
        if job_result.status.value == "completed" and job_result.output_path:
            response.result_url = f"/download/{job_id}"

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """
    Download translated PDF.

    - **job_id**: Job identifier
    """

    try:
        job_result = await job_manager.get_job_status(job_id)

        if not job_result:
            raise HTTPException(status_code=404, detail="Job not found")

        if job_result.status.value != "completed":
            raise HTTPException(
                status_code=400,
                detail="Translation not completed"
            )

        if not job_result.output_path:
            raise HTTPException(
                status_code=404,
                detail="Output file not found"
            )

        # Return file
        return FileResponse(
            path=job_result.output_path,
            media_type="application/pdf",
            filename=f"translated_{job_id}.pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download result: {str(e)}"
        )

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a translation job.

    - **job_id**: Job identifier
    """

    try:
        success = await job_manager.cancel_job(job_id)

        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

        return {"message": "Job cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )

@app.get("/jobs", response_model=List[JobStatusModel])
async def list_jobs(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None
):
    """
    List translation jobs.

    - **limit**: Maximum number of results
    - **offset**: Number of results to skip
    - **status**: Filter by status
    """

    try:
        # This is a simplified implementation
        # In production, you'd want to implement proper pagination
        jobs = []

        # For now, return empty list
        # In a real implementation, you'd query the job manager for active jobs
        return jobs

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""

    return {
        "source_languages": [
            {"code": "auto", "name": "Auto-detect"},
            {"code": "en", "name": "English"},
            {"code": "zh", "name": "Chinese"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ar", "name": "Arabic"},
            {"code": "ru", "name": "Russian"},
            {"code": "pt", "name": "Portuguese"}
        ],
        "target_languages": [
            {"code": "en", "name": "English"},
            {"code": "zh", "name": "Chinese"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ar", "name": "Arabic"},
            {"code": "ru", "name": "Russian"},
            {"code": "pt", "name": "Portuguese"}
        ]
    }

# Helper functions

def estimate_processing_time(file_size: int, document_type: str, use_vla: bool) -> int:
    """Estimate processing time in seconds"""

    # Base time per MB
    time_per_mb = 10  # seconds

    # Adjust for document type
    type_multipliers = {
        "general": 1.0,
        "scientific": 1.5,
        "legal": 1.2,
        "technical": 1.3
    }

    multiplier = type_multipliers.get(document_type, 1.0)

    # Adjust for VLA
    if use_vla:
        multiplier *= 2

    # Calculate
    size_mb = file_size / (1024 * 1024)
    estimated = int(size_mb * time_per_mb * multiplier)

    # Minimum 30 seconds
    return max(30, estimated)

def get_status_message(status: str, progress: int) -> str:
    """Get human-readable status message"""

    messages = {
        "pending": "Job is queued for processing",
        "processing": f"Processing... {progress}% complete",
        "extracting": f"Extracting content... {progress}% complete",
        "translating": f"Translating content... {progress}% complete",
        "reconstructing": f"Reconstructing PDF... {progress}% complete",
        "completed": "Translation completed successfully",
        "failed": "Translation failed",
        "cancelled": "Job was cancelled"
    }

    return messages.get(status, f"Status: {status}")

# Exception handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("PDF Translation Pipeline API starting...")
    # Initialize models, connections, etc.

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("PDF Translation Pipeline API shutting down...")
    # Close connections, save state, etc.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
