"""
Job Manager - Main orchestrator for the PDF translation pipeline.
Manages job lifecycle from submission to completion.
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime
import uuid
from pathlib import Path

from .schemas.job import TranslationJob, JobStatus, JobResult, TranslationRequest
from .extractors import (
    MarginManager, LayoutManager, FontExtractor,
    FormulaExtractor, TableExtractor, WatermarkExtractor
)
from .deciders import ContentDetector, VLATrigger, EdgeCaseHandler
from .xliff import XLIFFGenerator
from .translation import GeminiClient
from .reconstruction import PDFReconstructor

logger = logging.getLogger(__name__)


class JobManager:
    """
    Main orchestrator for the PDF translation pipeline.
    Manages job lifecycle from submission to completion.
    """

    def __init__(self):
        self.active_jobs: Dict[str, TranslationJob] = {}
        self.completed_jobs: Dict[str, JobResult] = {}

        # Initialize pipeline components
        self.margin_manager = MarginManager()
        self.layout_manager = LayoutManager()
        self.font_extractor = FontExtractor()
        self.formula_extractor = FormulaExtractor()
        self.table_extractor = TableExtractor()
        self.watermark_extractor = WatermarkExtractor()

        self.content_detector = ContentDetector()
        self.vla_trigger = VLATrigger()
        self.edge_case_handler = EdgeCaseHandler()

        self.xliff_generator = XLIFFGenerator()
        self.gemini_client = GeminiClient()
        self.pdf_reconstructor = PDFReconstructor()

        # Configuration
        self.upload_dir = Path("data/uploads")
        self.output_dir = Path("data/outputs")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def submit_job(self, filename: str, file_data: bytes,
                        mime_type: str, request: TranslationRequest) -> str:
        """
        Submit a new translation job.

        Args:
            filename: Original filename
            file_data: PDF file bytes
            mime_type: MIME type of the file
            request: Translation request parameters

        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        file_path = self.upload_dir / f"{job_id}.pdf"

        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(file_data)

        job = TranslationJob(
            id=job_id,
            filename=filename,
            file_path=str(file_path),
            file_size=len(file_data),
            mime_type=mime_type,
            request=request
        )

        self.active_jobs[job_id] = job
        logger.info(f"Submitted new job {job_id} for file {filename}")

        # Start processing in background
        asyncio.create_task(self._process_job(job))

        return job_id

    async def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """
        Get current status of a job.

        Args:
            job_id: Job identifier

        Returns:
            JobResult with current status, or None if not found
        """
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return JobResult(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                output_path=job.metadata.get('output_path'),
                error_message=job.error_message,
                metrics=job.metadata.get('metrics'),
                created_at=job.created_at,
                completed_at=job.completed_at
            )
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if successfully cancelled, False otherwise
        """
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                logger.info(f"Cancelled job {job_id}")
                return True
        return False

    async def _process_job(self, job: TranslationJob):
        """
        Main job processing pipeline.
        """
        try:
            logger.info(f"Starting job {job.id}")

            # Mark as started
            job.mark_started()
            job.update_progress("initializing", 0, "Starting PDF translation pipeline")

            # Phase 1: Extraction
            await self._extraction_phase(job)

            # Phase 2: Decision Making
            await self._decision_phase(job)

            # Phase 3: XLIFF Generation
            await self._xliff_phase(job)

            # Phase 4: Translation
            await self._translation_phase(job)

            # Phase 5: Reconstruction
            await self._reconstruction_phase(job)

            # Mark as completed
            job.mark_completed()
            logger.info(f"Completed job {job.id}")

            # Move to completed jobs
            result = JobResult(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                output_path=job.metadata.get('output_path'),
                metrics=job.metadata.get('metrics'),
                created_at=job.created_at,
                completed_at=job.completed_at
            )
            self.completed_jobs[job.id] = result
            del self.active_jobs[job.id]

        except Exception as e:
            logger.error(f"Job {job.id} failed: {str(e)}")
            job.mark_failed(str(e))
            result = JobResult(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                error_message=job.error_message,
                created_at=job.created_at,
                completed_at=job.completed_at
            )
            self.completed_jobs[job.id] = result
            del self.active_jobs[job.id]

    async def _extraction_phase(self, job: TranslationJob):
        """Phase 1: Extract document structure and content."""
        job.update_progress("extraction", 10, "Extracting document structure")

        # Extract margins
        job.metadata['margins'] = await self.margin_manager.extract_margins(job.file_path)

        # Extract layout information
        job.metadata['layout'] = await self.layout_manager.extract_layout(job.file_path)

        # Extract fonts
        job.metadata['fonts'] = await self.font_extractor.extract_fonts(job.file_path)

        # Extract formulas (if requested)
        if job.request.preserve_formulas:
            job.metadata['formulas'] = await self.formula_extractor.extract_formulas(job.file_path)

        # Extract tables (if requested)
        if job.request.preserve_tables:
            job.metadata['tables'] = await self.table_extractor.extract_tables(job.file_path)

        # Extract watermarks
        job.metadata['watermarks'] = await self.watermark_extractor.extract_watermarks(job.file_path)

        job.update_progress("extraction", 30, "Document structure extracted")

    async def _decision_phase(self, job: TranslationJob):
        """Phase 2: Make decisions about processing strategy."""
        job.update_progress("decision", 35, "Analyzing content for optimal processing")

        # Detect content types
        content_analysis = await self.content_detector.detect_content(job.file_path, job.metadata)

        # Determine if VLA is needed
        vla_needed = await self.vla_trigger.should_use_vla(content_analysis)

        # Handle edge cases
        edge_cases = await self.edge_case_handler.detect_edge_cases(content_analysis)

        job.metadata.update({
            'content_analysis': content_analysis,
            'vla_needed': vla_needed,
            'edge_cases': edge_cases
        })

        job.update_progress("decision", 40, "Processing strategy determined")

    async def _xliff_phase(self, job: TranslationJob):
        """Phase 3: Generate XLIFF for translation."""
        job.update_progress("xliff", 45, "Generating XLIFF document")

        xliff_data = await self.xliff_generator.generate_xliff(
            job.file_path,
            job.request,
            job.metadata
        )

        job.metadata['xliff'] = xliff_data
        job.update_progress("xliff", 60, "XLIFF document generated")

    async def _translation_phase(self, job: TranslationJob):
        """Phase 4: Translate content."""
        job.update_progress("translation", 65, "Translating document content")

        # Translate XLIFF content
        translated_xliff = await self.gemini_client.translate_xliff(
            job.metadata['xliff'],
            job.request.source_lang,
            job.request.target_lang
        )

        job.metadata['translated_xliff'] = translated_xliff
        job.update_progress("translation", 80, "Translation completed")

    async def _reconstruction_phase(self, job: TranslationJob):
        """Phase 5: Reconstruct PDF with translated content."""
        job.update_progress("reconstruction", 85, "Reconstructing PDF document")

        # Reconstruct PDF
        output_path = await self.pdf_reconstructor.reconstruct_pdf(
            job.file_path,
            job.metadata['translated_xliff'],
            job.metadata
        )

        # Calculate metrics
        metrics = await self._calculate_metrics(job)

        job.mark_completed(output_path, metrics)
        job.update_progress("reconstruction", 100, "PDF reconstruction completed")

    async def _calculate_metrics(self, job: TranslationJob) -> Dict[str, float]:
        """Calculate translation and layout preservation metrics."""
        # Placeholder for metrics calculation
        # In a real implementation, this would compare original vs translated content
        return {
            'translation_accuracy': 0.98,
            'layout_preservation': 0.95,
            'processing_time': (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
        }
