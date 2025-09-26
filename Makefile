.PHONY: api worker up down

api:
	uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	python -m worker.entrypoint

up:
	docker compose up -d --build

down:
	docker compose down -v
