-- PDF Translation Pipeline Database Setup
-- PostgreSQL database schema

-- Create database (run as superuser)
-- CREATE DATABASE pdf_translations;

-- Connect to the database
-- \c pdf_translations;

-- Create tables
CREATE TABLE IF NOT EXISTS translation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    source_lang VARCHAR(10),
    target_lang VARCHAR(10) NOT NULL,
    document_type VARCHAR(50) DEFAULT 'general',

    -- File information
    original_filename VARCHAR(255) NOT NULL,
    file_size INTEGER,
    page_count INTEGER,

    -- Paths
    input_path TEXT,
    output_path TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Progress tracking
    progress INTEGER DEFAULT 0,
    current_phase VARCHAR(50),

    -- Results
    error_message TEXT,
    metrics JSONB,

    -- User information (if implementing auth)
    user_id UUID,
    api_key VARCHAR(255)
);

-- Create indexes for translation_jobs
CREATE INDEX IF NOT EXISTS idx_status ON translation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_created_at ON translation_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_user_id ON translation_jobs(user_id);

-- Translation memory table
CREATE TABLE IF NOT EXISTS translation_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_text TEXT NOT NULL,
    target_text TEXT NOT NULL,
    source_lang VARCHAR(10) NOT NULL,
    target_lang VARCHAR(10) NOT NULL,
    document_type VARCHAR(50),
    confidence FLOAT,

    -- Metadata
    job_id UUID REFERENCES translation_jobs(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    used_count INTEGER DEFAULT 0,
    last_used TIMESTAMP WITH TIME ZONE
);

-- Create indexes for translation_memory
CREATE INDEX IF NOT EXISTS idx_source_hash ON translation_memory(MD5(source_text));
CREATE INDEX IF NOT EXISTS idx_langs ON translation_memory(source_lang, target_lang);

-- Document metadata table
CREATE TABLE IF NOT EXISTS document_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES translation_jobs(id) ON DELETE CASCADE,
    
    -- Document properties
    title TEXT,
    author TEXT,
    subject TEXT,
    keywords TEXT[],
    creation_date TIMESTAMP,
    modification_date TIMESTAMP,
    
    -- Layout information
    margins JSONB,
    fonts JSONB,
    layout_complexity VARCHAR(20),
    
    -- Special elements
    has_tables BOOLEAN DEFAULT FALSE,
    has_formulas BOOLEAN DEFAULT FALSE,
    has_images BOOLEAN DEFAULT FALSE,
    has_watermarks BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Processing metrics table
CREATE TABLE IF NOT EXISTS processing_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES translation_jobs(id) ON DELETE CASCADE,
    
    -- Timing metrics (in seconds)
    extraction_time FLOAT,
    translation_time FLOAT,
    reconstruction_time FLOAT,
    total_time FLOAT,
    
    -- Quality metrics
    translation_score FLOAT,
    layout_preservation_score FLOAT,
    
    -- Resource usage
    memory_used_mb INTEGER,
    cpu_percent FLOAT,
    gpu_used BOOLEAN DEFAULT FALSE,
    
    -- Counts
    total_segments INTEGER,
    cached_segments INTEGER,
    vla_triggered BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API keys table (if implementing authentication)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100),
    
    -- Permissions
    is_active BOOLEAN DEFAULT TRUE,
    rate_limit INTEGER DEFAULT 100,
    
    -- Usage tracking
    total_requests INTEGER DEFAULT 0,
    total_pages_processed INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Create views for monitoring

-- Active jobs view
CREATE OR REPLACE VIEW active_jobs AS
SELECT 
    id,
    status,
    original_filename,
    source_lang,
    target_lang,
    progress,
    current_phase,
    created_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) as seconds_elapsed
FROM translation_jobs
WHERE status IN ('pending', 'processing', 'extracting', 'translating', 'reconstructing')
ORDER BY created_at DESC;

-- Job statistics view
CREATE OR REPLACE VIEW job_statistics AS
SELECT 
    COUNT(*) as total_jobs,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_jobs,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_jobs,
    AVG(CASE WHEN status = 'completed' THEN page_count END) as avg_pages,
    AVG(CASE WHEN status = 'completed' THEN 
        EXTRACT(EPOCH FROM (completed_at - created_at)) 
    END) as avg_processing_time_seconds
FROM translation_jobs
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';

-- Create functions

-- Function to clean old jobs
CREATE OR REPLACE FUNCTION clean_old_jobs() RETURNS void AS $$
BEGIN
    DELETE FROM translation_jobs 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
    AND status IN ('completed', 'failed', 'cancelled');
END;
$$ LANGUAGE plpgsql;

-- Function to update job progress
CREATE OR REPLACE FUNCTION update_job_progress(
    p_job_id UUID,
    p_progress INTEGER,
    p_phase VARCHAR(50)
) RETURNS void AS $$
BEGIN
    UPDATE translation_jobs
    SET 
        progress = p_progress,
        current_phase = p_phase,
        started_at = CASE 
            WHEN started_at IS NULL THEN CURRENT_TIMESTAMP 
            ELSE started_at 
        END
    WHERE id = p_job_id;
END;
$$ LANGUAGE plpgsql;

-- Job pages table (for tracking individual page processing)
CREATE TABLE IF NOT EXISTS job_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES translation_jobs(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    -- Processing info
    extracted_text TEXT,
    translated_text TEXT,
    layout_data JSONB,

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Errors
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_job_pages_job_id ON job_pages(job_id);
CREATE INDEX IF NOT EXISTS idx_job_pages_status ON job_pages(status);

-- Translation cache table (for caching translation results)
CREATE TABLE IF NOT EXISTS translation_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_text_hash VARCHAR(64) NOT NULL,
    source_lang VARCHAR(10) NOT NULL,
    target_lang VARCHAR(10) NOT NULL,

    -- Cached translation
    translated_text TEXT NOT NULL,
    confidence FLOAT,

    -- Metadata
    model_used VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,

    -- TTL
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_translation_cache_hash ON translation_cache(source_text_hash, source_lang, target_lang);
CREATE INDEX IF NOT EXISTS idx_translation_cache_expires ON translation_cache(expires_at);

-- Error logs table (for tracking system errors)
CREATE TABLE IF NOT EXISTS error_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES translation_jobs(id) ON DELETE SET NULL,

    -- Error details
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,

    -- Context
    component VARCHAR(100),
    severity VARCHAR(20) DEFAULT 'error',
    metadata JSONB,

    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,

    -- Resolution
    resolution_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_error_logs_job_id ON error_logs(job_id);
CREATE INDEX IF NOT EXISTS idx_error_logs_type ON error_logs(error_type);
CREATE INDEX IF NOT EXISTS idx_error_logs_created ON error_logs(created_at);

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO translator;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO translator;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO translator;