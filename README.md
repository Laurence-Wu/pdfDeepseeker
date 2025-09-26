# pdfDeepseeker

A high-performance PDF translation service that preserves formatting while extracting, translating, and reassembling PDF documents.

## Architecture

This project uses a modern microservice architecture with:

- **FastAPI**: High-performance async API framework
- **Redis + RQ**: Scalable job queue and worker system
- **PostgreSQL**: Robust database for job tracking and metadata
- **Python**: Primary language with clean, maintainable code

## Project Structure

```
pdfDeepseeker/
├── api/                    # FastAPI application
│   └── app/
│       ├── config.py       # Application configuration
│       ├── db.py          # Database connection and session management
│       ├── redis.py       # Redis client configuration
│       ├── main.py        # FastAPI application entry point
│       └── routes/        # API endpoints
│           ├── health.py  # Health check endpoint
│           └── jobs.py    # Job submission and status endpoints
├── worker/                # Background job processing
│   ├── tasks.py          # Task definitions
│   └── entrypoint.py     # Worker process entry point
├── libs/                  # Shared libraries and utilities
│   ├── adapters/         # External service adapters
│   │   ├── gemini/       # Google Gemini AI integration
│   │   └── pdf/          # PDF processing utilities
│   └── middlelang/       # Intermediate format schemas
├── configs/              # Configuration files
├── scripts/              # Development and deployment scripts
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Local development environment
└── Makefile             # Common development commands
```

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Laurence-Wu/pdfDeepseeker.git
   cd pdfDeepseeker
   ```

2. **Set up environment:**
   ```bash
   cp configs/.env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start services:**
   ```bash
   make up  # or: docker compose up -d
   ```

5. **Run the API:**
   ```bash
   make api  # or: uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Start a worker:**
   ```bash
   make worker  # or: python -m worker.entrypoint
   ```

## API Endpoints

- `GET /health` - Health check
- `POST /jobs/` - Submit a translation job

## Development

- **API**: `make api` - Start the API server with auto-reload
- **Worker**: `make worker` - Start the background worker
- **Docker**: `make up` - Start all services with Docker Compose
- **Clean**: `make down` - Stop all services and clean up

## Next Steps

This scaffold provides the foundation for a PDF translation service. Future development will include:

- PDF extraction and parsing
- Integration with translation services
- Format-preserving PDF reconstruction
- Advanced job management and monitoring
- Scalable deployment configurations
