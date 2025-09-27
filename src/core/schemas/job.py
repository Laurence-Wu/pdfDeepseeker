from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    TRANSLATING = "translating"
    RECONSTRUCTING = "reconstructing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TranslationRequest(BaseModel):
    source_lang: str = Field(..., min_length=2, max_length=10)
    target_lang: str = Field(..., min_length=2, max_length=10)
    preserve_formatting: bool = True
    preserve_formulas: bool = True
    preserve_tables: bool = True


class JobProgress(BaseModel):
    stage: str
    progress: float  # 0-100
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    progress: List[JobProgress]
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class TranslationJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str
    file_size: int
    mime_type: str
    request: TranslationRequest
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    progress: List[JobProgress] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def update_progress(self, stage: str, progress: float, message: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None):
        """Update job progress with new stage information."""
        self.progress.append(JobProgress(
            stage=stage,
            progress=progress,
            message=message,
            details=details
        ))

    def mark_started(self):
        """Mark job as started."""
        self.status = JobStatus.PROCESSING
        self.started_at = datetime.utcnow()

    def mark_completed(self, output_path: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None):
        """Mark job as completed successfully."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if output_path:
            self.metadata['output_path'] = output_path
        if metrics:
            self.metadata['metrics'] = metrics

    def mark_failed(self, error_message: str):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
