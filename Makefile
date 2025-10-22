.PHONY: api worker up down test install

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	python -m src.worker.entrypoint

up:
	docker compose up -d --build

down:
	docker compose down -v

test:
	pytest tests/

install:
	pip install -r requirements.txt
