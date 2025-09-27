from fastapi import FastAPI
from .routes import health, jobs

app = FastAPI(title="pdfDeepseeker API")
app.include_router(health.router)
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

