-- PostgreSQL Schema for Sarah AI CRM
-- Complete database schema for the multi-domain WebUI system

-- Create database and user (run as postgres superuser)
-- CREATE DATABASE sarah_ai_crm;
-- CREATE USER sarah_ai WITH PASSWORD 'your-secure-password-here';
-- GRANT ALL PRIVILEGES ON DATABASE sarah_ai_crm TO sarah_ai;

-- Connect to sarah_ai_crm database and run the following:

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('admin', 'user', 'viewer')),
    domain VARCHAR(50) NOT NULL,
    permissions TEXT[] DEFAULT ARRAY[]::TEXT[],
    is_active BOOLEAN DEFAULT true,
    is_system BOOLEAN DEFAULT false,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Companies table
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(50) NOT NULL,
    industry VARCHAR(100),
    size_category VARCHAR(50),
    website VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Campaigns table
CREATE TABLE campaigns (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(50) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'draft')),
    script_template TEXT,
    voice_settings JSONB DEFAULT '{}',
    call_schedule JSONB DEFAULT '{}',
    created_by INTEGER REFERENCES users(id),
    target_count INTEGER DEFAULT 0,
    completed_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Leads table
CREATE TABLE leads (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(20) NOT NULL,
    company_id INTEGER REFERENCES companies(id),
    company VARCHAR(255),
    title VARCHAR(100),
    domain VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'new' CHECK (status IN ('new', 'contacted', 'qualified', 'proposal', 'negotiation', 'closed_won', 'closed_lost', 'imported')),
    priority VARCHAR(10) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    source VARCHAR(50) DEFAULT 'unknown',
    assigned_to INTEGER REFERENCES users(id),
    campaign_id INTEGER REFERENCES campaigns(id),
    estimated_value DECIMAL(10,2),
    call_count INTEGER DEFAULT 0,
    last_contacted TIMESTAMP,
    next_follow_up TIMESTAMP,
    notes TEXT,
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    custom_fields JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Call logs table
CREATE TABLE call_logs (
    id SERIAL PRIMARY KEY,
    lead_id INTEGER REFERENCES leads(id),
    campaign_id INTEGER REFERENCES campaigns(id),
    user_id INTEGER REFERENCES users(id),
    ai_server_id VARCHAR(100),
    phone_number VARCHAR(20) NOT NULL,
    call_sid VARCHAR(100),
    direction VARCHAR(10) DEFAULT 'outbound' CHECK (direction IN ('inbound', 'outbound')),
    status VARCHAR(20),
    outcome VARCHAR(50),
    duration INTEGER, -- in seconds
    recording_url VARCHAR(500),
    ai_summary TEXT,
    ai_confidence DECIMAL(3,2),
    objections_handled TEXT[] DEFAULT ARRAY[]::TEXT[],
    interest_level VARCHAR(20),
    next_action VARCHAR(100),
    scheduled_follow_up TIMESTAMP,
    voice_engine VARCHAR(50),
    ai_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Activities table for tracking lead interactions
CREATE TABLE activities (
    id SERIAL PRIMARY KEY,
    lead_id INTEGER REFERENCES leads(id),
    user_id INTEGER REFERENCES users(id),
    activity_type VARCHAR(50) NOT NULL,
    subject VARCHAR(255),
    description TEXT,
    completed_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- AI deployments table (Star AI feature)
CREATE TABLE ai_deployments (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    vast_instance_id INTEGER,
    tier VARCHAR(50) NOT NULL CHECK (tier IN ('starter', 'professional', 'enterprise')),
    domain VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'deploying' CHECK (status IN ('deploying', 'starting', 'ready', 'error', 'terminated')),
    public_ip VARCHAR(45),
    endpoints JSONB DEFAULT '{}',
    cost_per_hour DECIMAL(10,4),
    total_cost DECIMAL(10,2) DEFAULT 0.00,
    gpu_info VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    terminated_at TIMESTAMP
);

-- Voice clones table
CREATE TABLE voice_clones (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id INTEGER REFERENCES users(id),
    domain VARCHAR(50) NOT NULL,
    voice_engine VARCHAR(50) NOT NULL CHECK (voice_engine IN ('f5_tts', 'riva', 'coqui')),
    model_path VARCHAR(500),
    sample_audio_url VARCHAR(500),
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- System settings table
CREATE TABLE system_settings (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value JSONB,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(domain, key)
);

-- Create indexes for performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_domain ON users(domain);
CREATE INDEX idx_leads_domain ON leads(domain);
CREATE INDEX idx_leads_status ON leads(status);
CREATE INDEX idx_leads_assigned_to ON leads(assigned_to);
CREATE INDEX idx_leads_phone ON leads(phone);
CREATE INDEX idx_call_logs_lead_id ON call_logs(lead_id);
CREATE INDEX idx_call_logs_created_at ON call_logs(created_at);
CREATE INDEX idx_campaigns_domain ON campaigns(domain);
CREATE INDEX idx_ai_deployments_user_id ON ai_deployments(user_id);
CREATE INDEX idx_ai_deployments_status ON ai_deployments(status);
CREATE INDEX idx_activities_lead_id ON activities(lead_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_companies_updated_at BEFORE UPDATE ON companies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_leads_updated_at BEFORE UPDATE ON leads FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_voice_clones_updated_at BEFORE UPDATE ON voice_clones FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_settings_updated_at BEFORE UPDATE ON system_settings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default system users for each domain
INSERT INTO users (username, email, password_hash, role, domain, permissions, is_system) VALUES
('system_smrt', 'system@smrtpayments.com', '$2b$12$placeholder_hash_here', 'admin', 'smrtpayments.com', ARRAY['*'], true),
('system_kjo', 'system@kjo.ai', '$2b$12$placeholder_hash_here', 'admin', 'kjo.ai', ARRAY['*'], true);

-- Insert default system settings
INSERT INTO system_settings (domain, key, value, description) VALUES
('smrtpayments.com', 'voice_engine', '"f5_tts"', 'Primary voice synthesis engine'),
('smrtpayments.com', 'max_concurrent_calls', '10', 'Maximum concurrent outbound calls'),
('smrtpayments.com', 'call_hours', '{"start": "09:00", "end": "17:00", "timezone": "UTC"}', 'Allowed calling hours'),
('kjo.ai', 'voice_engine', '"f5_tts"', 'Primary voice synthesis engine'),
('kjo.ai', 'max_concurrent_calls', '5', 'Maximum concurrent outbound calls'),
('kjo.ai', 'call_hours', '{"start": "09:00", "end": "17:00", "timezone": "UTC"}', 'Allowed calling hours');

-- Grant permissions to sarah_ai user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sarah_ai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sarah_ai;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO sarah_ai;

-- Enable row level security (optional, for multi-tenancy)
-- ALTER TABLE leads ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE campaigns ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE call_logs ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (uncomment if using RLS)
-- CREATE POLICY domain_isolation_leads ON leads FOR ALL TO sarah_ai USING (domain = current_setting('app.current_domain'));
-- CREATE POLICY domain_isolation_campaigns ON campaigns FOR ALL TO sarah_ai USING (domain = current_setting('app.current_domain'));