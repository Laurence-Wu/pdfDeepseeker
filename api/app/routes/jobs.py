from fastapi import APIRouter
from rq import Queue
from ..redis import redis_client

router = APIRouter()
q = Queue("default", connection=redis_client)

@router.post("/")
def create_job():
    job = q.enqueue("worker.tasks.example_task", 1, 2)
    return {"job_id": job.get_id()}
