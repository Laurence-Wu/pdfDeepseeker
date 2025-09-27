# PDF Translation Pipeline - Configuration Guide

## Environment Configuration

### .env File
```bash
# Environment
ENVIRONMENT=development

# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=translator
DB_PASSWORD=secure_password
DB_NAME=pdf_translations

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# OpenRouter Configuration (for Gemini in China)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemini-pro-1.5
OPENROUTER_TIMEOUT=60

# Alternative Models (via OpenRouter)
FALLBACK_MODEL=anthropic/claude-3-opus
SECONDARY_MODEL=meta-llama/llama-3.1-70b

# GPU Support
USE_GPU=false
GPU_DEVICE=cuda:0
GPU_MEMORY_LIMIT=8192

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
MAX_WORKERS=10
MAX_PAGES_PER_JOB=2000
BATCH_SIZE=5
PARALLEL_PAGES=10

# Cache
CACHE_TTL=3600
MAX_CACHE_SIZE=10GB

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
```

## Main Configuration (config.yaml)

```yaml
pipeline:
  name: "PDF Translation Pipeline"
  version: "2.0.0"
  environment: ${ENVIRONMENT}

# OpenRouter Configuration for Gemini
translation:
  primary_service:
    provider: "openrouter"
    api_key: ${OPENROUTER_API_KEY}
    base_url: ${OPENROUTER_BASE_URL}
    model: ${OPENROUTER_MODEL}
    parameters:
      temperature: 0.3
      top_p: 0.9
      max_tokens: 2048
      timeout: 60
    retry:
      max_attempts: 3
      backoff_factor: 2
      max_delay: 30
  
  fallback_service:
    provider: "openrouter"
    model: ${FALLBACK_MODEL}
    enabled: true

# Extraction Settings
extraction:
  margins:
    enabled: true
    method: "pdfplumber"
    threshold: 10
    enforce_strict: true
  
  fonts:
    extract_embedded: true
    cache_fonts: true
    fallback_chain: ["Arial", "Times New Roman", "Helvetica"]
  
  layout:
    model: "PubLayNet"
    confidence_threshold: 0.8
    column_detection: true
    relationship_detection: true
  
  formulas:
    model: "LaTeX-OCR"
    confidence_threshold: 0.85
    preserve_as_image: false
    extract_latex: true
  
  tables:
    model: "table-transformer"
    confidence_threshold: 0.75
    extract_structure: true
  
  edge_cases:
    detect_all: true
    handlers:
      rotated_text: true
      vertical_text: true
      footnotes: true
      drop_caps: true
      form_fields: true
      annotations: true
      hyperlinks: true

# VLA Configuration
vla:
  enabled: true
  trigger_threshold: 0.7
  models:
    primary:
      name: "surya"
      repo: "VikParuchuri/surya"
      use_gpu: ${USE_GPU}
    fallback:
      name: "paddleocr"
      use_angle_cls: true
  complexity_thresholds:
    simple: 0.3
    moderate: 0.5
    complex: 0.7
    extreme: 0.9

# Text Control
text_control:
  enabled: true
  max_expansion_ratio: 1.1
  min_compression_ratio: 0.7
  adjustment_strategies:
    - abbreviation
    - spacing_adjustment
    - font_size_reduction
  retry_on_overflow: true
  max_retries: 3

# Reconstruction
reconstruction:
  preserve_margins: true
  preserve_layout: true
  embed_fonts: true
  quality:
    image_dpi: 300
    compression: "lossless"
    pdf_version: "1.7"

# Processing Limits
limits:
  max_file_size: 524288000  # 500MB
  max_pages: ${MAX_PAGES_PER_JOB}
  timeout:
    extraction: 300
    translation: 600
    reconstruction: 300
  
# Cache Configuration
cache:
  redis:
    enabled: true
    ttl: ${CACHE_TTL}
    max_size: ${MAX_CACHE_SIZE}
  
  memory:
    enabled: true
    max_size: "1GB"
    eviction_policy: "lru"
  
  disk:
    enabled: false
    path: "/tmp/pdf_cache"
    max_size: "50GB"

# Monitoring
monitoring:
  enabled: ${METRICS_ENABLED}
  port: ${METRICS_PORT}
  collectors:
    - process_metrics
    - translation_quality
    - processing_speed
    - error_rates
  
  alerts:
    error_rate_threshold: 0.05
    response_time_threshold: 30
    queue_depth_threshold: 1000
```

## Model-Specific Configurations

### VLA Models Configuration (vla_models.yaml)
```yaml
models:
  surya:
    repo: "VikParuchuri/surya"
    version: "latest"
    requirements:
      gpu_memory: 4096
      cpu_memory: 8192
    settings:
      batch_size: 8
      confidence_threshold: 0.85
  
  mplug_docowl:
    repo: "mPLUG/DocOwl"
    version: "1.5"
    requirements:
      gpu_memory: 8192
      cpu_memory: 16384
    settings:
      max_resolution: 4096
      use_for_extreme_cases: true
  
  layoutlm:
    repo: "microsoft/layoutlmv3-base"
    version: "latest"
    requirements:
      gpu_memory: 2048
      cpu_memory: 4096
    settings:
      max_sequence_length: 512
  
  paddleocr:
    repo: "PaddlePaddle/PaddleOCR"
    version: "2.7.0"
    settings:
      use_angle_cls: true
      lang: "ch"
      det_db_thresh: 0.3
      rec_batch_num: 6
```

### Translation Prompts Configuration (prompts.yaml)
```yaml
prompts:
  system:
    default: |
      You are a professional document translator preserving exact formatting.
      Maintain the same character count when possible.
      Never translate formulas, code, or technical identifiers.
  
  document_types:
    scientific:
      context: "Scientific paper with technical terminology"
      preserve: ["formulas", "citations", "figures"]
    
    legal:
      context: "Legal document requiring precise terminology"
      preserve: ["clause_numbers", "legal_terms", "formatting"]
    
    technical:
      context: "Technical manual with specifications"
      preserve: ["measurements", "part_numbers", "diagrams"]
  
  constraints:
    length_strict: |
      CRITICAL: Output must be ≤{max_chars} characters.
      If impossible, abbreviate non-critical words.
    
    layout_preserve: |
      Maintain exact line breaks at: {line_breaks}
      Text must fit bbox: {bbox_dimensions}
```

## Resource Allocation

### Docker Compose Configuration
```yaml
version: '3.8'

services:
  api:
    image: pdf-translator:latest
    environment:
      - ENVIRONMENT=production
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
  
  extraction-worker:
    image: pdf-translator:latest
    command: worker extraction
    deploy:
      replicas: 3
    resources:
      limits:
        cpus: '2'
        memory: 8G
  
  translation-worker:
    image: pdf-translator:latest
    command: worker translation
    deploy:
      replicas: 2
    resources:
      limits:
        cpus: '1'
        memory: 4G
  
  vla-processor:
    image: pdf-translator:latest
    command: worker vla
    deploy:
      replicas: 1
    resources:
      limits:
        cpus: '4'
        memory: 16G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Deployment Configurations

### Production Settings
```yaml
production:
  debug: false
  workers: 10
  log_level: WARNING
  cache_aggressive: true
  monitoring_detailed: true
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
  
  health_checks:
    enabled: true
    interval: 30
    timeout: 5
    failure_threshold: 3
```

### Development Settings
```yaml
development:
  debug: true
  workers: 2
  log_level: DEBUG
  cache_aggressive: false
  monitoring_detailed: false
  hot_reload: true
  mock_services:
    enabled: true
    services: ["translation"]
```
