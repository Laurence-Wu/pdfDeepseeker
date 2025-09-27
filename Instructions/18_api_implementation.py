# FastAPI Service Implementation - PDF Translation Pipeline

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import asyncio
from datetime import datetime
import os
from pathlib import Path

# Import pipeline components
from integrated_pipeline import IntegratedPDFTranslationPipeline
from celery import Celery
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

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
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://translator:password@localhost/pdf_translations")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
settings = Settings()

# Create directories
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize services
redis_client = redis.from_url(settings.REDIS_URL)
celery_app = Celery("pdf_translator", broker=settings.REDIS_URL)

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Pipeline instance
pipeline = IntegratedPDFTranslationPipeline()

# Pydantic models
class TranslationRequest(BaseModel):
    """Translation job request"""
    source_lang: str = Field(default="auto", description="Source language code or 'auto'")
    target_lang: str = Field(..., description="Target language code")
    document_type: Optional[str] = Field(default="general", description="Document type")
    preserve_formatting: bool = Field(default=True, description="Preserve formatting")
    use_vla: Optional[bool] = Field(default=None, description="Force VLA usage")
    priority: int = Field(default=5, description="Priority (1-10)")
    callback_url: Optional[str] = Field(default=None, description="Webhook for completion")

class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: int
    message: Optional[str]
    created_at: datetime
    updated_at: datetime
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

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
        "redis": redis_client.ping(),
        "database": check_database_health(),
        "workers": check_workers_health(),
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
    request: TranslationRequest = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
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
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = settings.UPLOAD_DIR / f"{job_id}.pdf"
    with open(upload_path, "wb") as f:
        f.write(contents)
    
    # Create job record in database
    job_record = create_job_record(
        db,
        job_id=job_id,
        filename=file.filename,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        document_type=request.document_type,
        file_size=file_size
    )
    
    # Queue translation task
    task = celery_app.send_task(
        'worker.tasks.translate_pdf',
        args=[str(upload_path), job_id, request.dict()],
        priority=request.priority
    )
    
    # Store task ID in Redis
    redis_client.set(f"job:{job_id}:task", task.id, ex=86400)  # 24 hours
    redis_client.set(f"job:{job_id}:status", "queued", ex=86400)
    
    # Schedule callback if provided
    if request.callback_url:
        background_tasks.add_task(
            schedule_callback,
            job_id,
            request.callback_url
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

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get translation job status.
    
    - **job_id**: Job identifier from translation request
    """
    
    # Get status from Redis
    status = redis_client.get(f"job:{job_id}:status")
    
    if not status:
        # Check database
        job = get_job_from_db(db, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        status = job.status
    else:
        status = status.decode('utf-8')
    
    # Get progress
    progress = redis_client.get(f"job:{job_id}:progress")
    progress = int(progress) if progress else 0
    
    # Get job details from database
    job = get_job_from_db(db, job_id)
    
    # Build response
    response = JobStatus(
        job_id=job_id,
        status=status,
        progress=progress,
        message=get_status_message(status, progress),
        created_at=job.created_at,
        updated_at=job.updated_at,
        result_url=None,
        error=None,
        metrics=None
    )
    
    # Add result URL if completed
    if status == "completed":
        output_file = settings.OUTPUT_DIR / f"{job_id}.pdf"
        if output_file.exists():
            response.result_url = f"/download/{job_id}"
            
        # Get metrics
        metrics = redis_client.get(f"job:{job_id}:metrics")
        if metrics:
            response.metrics = json.loads(metrics)
    
    # Add error if failed
    elif status == "failed":
        error = redis_client.get(f"job:{job_id}:error")
        if error:
            response.error = error.decode('utf-8')
    
    return response

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """
    Download translated PDF.
    
    - **job_id**: Job identifier
    """
    
    # Check if job is completed
    status = redis_client.get(f"job:{job_id}:status")
    if not status or status.decode('utf-8') != "completed":
        raise HTTPException(
            status_code=400,
            detail="Translation not completed or not found"
        )
    
    # Check if output file exists
    output_file = settings.OUTPUT_DIR / f"{job_id}.pdf"
    if not output_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Output file not found"
        )
    
    # Return file
    return FileResponse(
        path=output_file,
        media_type="application/pdf",
        filename=f"translated_{job_id}.pdf"
    )

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a translation job.
    
    - **job_id**: Job identifier
    """
    
    # Get task ID
    task_id = redis_client.get(f"job:{job_id}:task")
    
    if not task_id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Cancel task
    celery_app.control.revoke(task_id.decode('utf-8'), terminate=True)
    
    # Update status
    redis_client.set(f"job:{job_id}:status", "cancelled", ex=86400)
    
    return {"message": "Job cancelled successfully"}

@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List translation jobs.
    
    - **limit**: Maximum number of results
    - **offset**: Number of results to skip
    - **status**: Filter by status
    """
    
    jobs = get_jobs_from_db(db, limit, offset, status)
    
    result = []
    for job in jobs:
        # Get current status from Redis
        redis_status = redis_client.get(f"job:{job.id}:status")
        current_status = redis_status.decode('utf-8') if redis_status else job.status
        
        progress = redis_client.get(f"job:{job.id}:progress")
        progress = int(progress) if progress else 0
        
        result.append(JobStatus(
            job_id=job.id,
            status=current_status,
            progress=progress,
            message=get_status_message(current_status, progress),
            created_at=job.created_at,
            updated_at=job.updated_at,
            result_url=f"/download/{job.id}" if current_status == "completed" else None,
            error=None,
            metrics=None
        ))
    
    return result

@app.get("/stats")
async def get_statistics():
    """Get service statistics"""
    
    stats = {
        "total_jobs": redis_client.get("stats:total_jobs") or 0,
        "completed_jobs": redis_client.get("stats:completed_jobs") or 0,
        "failed_jobs": redis_client.get("stats:failed_jobs") or 0,
        "active_jobs": redis_client.get("stats:active_jobs") or 0,
        "average_processing_time": redis_client.get("stats:avg_time") or 0,
        "total_pages_processed": redis_client.get("stats:total_pages") or 0,
    }
    
    # Convert byte strings to integers
    for key in stats:
        if isinstance(stats[key], bytes):
            stats[key] = int(stats[key])
    
    return stats

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

# WebSocket for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket
    
    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]
    
    async def send_update(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Send updates every second
            await asyncio.sleep(1)
            
            # Get current status
            status = redis_client.get(f"job:{job_id}:status")
            progress = redis_client.get(f"job:{job_id}:progress")
            
            if status:
                await manager.send_update(job_id, {
                    "status": status.decode('utf-8'),
                    "progress": int(progress) if progress else 0,
                    "message": get_status_message(
                        status.decode('utf-8'),
                        int(progress) if progress else 0
                    )
                })
                
                # Close connection if job is done
                if status.decode('utf-8') in ['completed', 'failed', 'cancelled']:
                    break
            
    except WebSocketDisconnect:
        manager.disconnect(job_id)

# Helper functions

def check_database_health() -> bool:
    """Check if database is accessible"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except:
        return False

def check_workers_health() -> bool:
    """Check if Celery workers are available"""
    try:
        i = celery_app.control.inspect()
        stats = i.stats()
        return stats is not None and len(stats) > 0
    except:
        return False

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
        "queued": "Job is queued for processing",
        "processing": f"Processing... {progress}% complete",
        "completed": "Translation completed successfully",
        "failed": "Translation failed",
        "cancelled": "Job was cancelled"
    }
    
    return messages.get(status, f"Status: {status}")

def create_job_record(db: Session, **kwargs):
    """Create job record in database"""
    # Implementation depends on your database schema
    pass

def get_job_from_db(db: Session, job_id: str):
    """Get job from database"""
    # Implementation depends on your database schema
    pass

def get_jobs_from_db(db: Session, limit: int, offset: int, status: Optional[str]):
    """Get jobs from database"""
    # Implementation depends on your database schema
    pass

async def schedule_callback(job_id: str, callback_url: str):
    """Schedule webhook callback on completion"""
    # Implementation for webhook callback
    pass

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