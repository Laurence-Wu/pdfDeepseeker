# PDF Translation Pipeline - Complete Architecture

## System Overview
A high-precision PDF translation pipeline achieving 98%+ accuracy with complete preservation of layout, fonts, margins, and all document elements.

## Core Architecture Layers

### 1. Input Layer
- **API Gateway**: REST API with FastAPI
- **Queue System**: RabbitMQ/Redis Queue for job management
- **Job Manager**: Orchestrates processing workflow

### 2. Extraction Layer
**Components:**
- `MarginManager`: Detects and enforces document margins
- `LayoutManager`: Analyzes spatial relationships between elements
- `FontExtractor`: Extracts embedded fonts and style information
- `FormulaExtractor`: Detects and extracts mathematical formulas
- `TableExtractor`: Identifies and structures tables
- `WatermarkExtractor`: Detects visible and invisible watermarks

### 3. Decision Layer
**Components:**
- `ContentDetector`: Routes content to appropriate processors
- `VLATrigger`: Determines when to use Vision-Language models
- `EdgeCaseHandler`: Manages special formatting cases

### 4. XLIFF Layer
**Components:**
- `XLIFFGenerator`: Creates XLIFF 2.1 documents
- `MetadataManager`: Handles all document metadata
- `SkeletonBuilder`: Builds layout reconstruction data

### 5. Translation Layer
**Components:**
- `GeminiClient`: Interfaces with Gemini via OpenRouter (China-compatible)
- `PromptEngine`: Generates context-aware prompts
- `TextLengthController`: Manages translation length constraints

### 6. Reconstruction Layer
**Components:**
- `PDFReconstructor`: Rebuilds the PDF document
- `LayoutRestorer`: Restores original layout
- `StyleApplier`: Applies fonts and styles

## Data Flow Architecture

```
Input PDF → Queue → Extraction → Decision → XLIFF Generation → Translation → Validation → Reconstruction → Output PDF
```

## Service Communication

### Synchronous Operations
- API ↔ Queue submission
- Validation checks
- Final output delivery

### Asynchronous Operations
- Queue → Worker processing
- Parallel page processing
- Translation batching

### Caching Strategy
- **L1 Cache**: In-memory (Worker) - 100MB
- **L2 Cache**: Redis - 10GB
- **L3 Cache**: PostgreSQL - Translation memory

## Scaling Architecture

### Horizontal Scaling
- Extraction workers: 1-20 instances
- Translation workers: 1-10 instances
- VLA processors: 1-5 instances

### Vertical Scaling
- Reconstruction service: Memory-intensive (8-16GB)
- VLA models: GPU-enabled instances

## Error Handling

### Retry Strategy
- Extraction failures: 3 retries with exponential backoff
- Translation failures: Fallback to simpler prompts
- VLA failures: Fallback to traditional OCR

### Fallback Mechanisms
- Primary: Gemini via OpenRouter
- Secondary: Traditional OCR + rule-based extraction
- Tertiary: Preserve original content

## Monitoring & Observability

### Metrics
- Translation accuracy (BLEU/BERT scores)
- Processing speed (pages/minute)
- Layout preservation (IoU scores)
- Resource utilization

### Logging
- Structured JSON logging
- Distributed tracing with correlation IDs
- Error aggregation and alerting

## Security

### Data Protection
- Encryption at rest and in transit
- PII detection and masking
- Secure file handling

### Access Control
- API key authentication
- Rate limiting
- IP whitelisting for production

## Performance Targets

- **Accuracy**: 98%+ translation quality
- **Layout Preservation**: 95%+ IoU score
- **Processing Speed**: 10 pages/minute
- **Concurrency**: 100 simultaneous jobs
- **Availability**: 99.9% uptime
