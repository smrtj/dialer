#!/usr/bin/env python3
"""
PostgreSQL Integration for Sarah AI WebUI
Direct PostgreSQL database integration replacing Supabase
"""
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
import asyncpg
import psycopg2
from contextlib import asynccontextmanager
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLIntegration:
    def __init__(self):
        """Initialize PostgreSQL client"""
        # Database connection settings
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.database = os.getenv("POSTGRES_DATABASE", "sarah_ai_crm")
        self.username = os.getenv("POSTGRES_USER", "sarah_ai")
        self.password = os.getenv("POSTGRES_PASSWORD", "your-password")
        
        # Connection pool
        self.pool = None
        
        # Connection string for sync operations
        self.sync_connection_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        # Async connection string
        self.async_connection_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        logger.info("PostgreSQL integration initialized")
    
    async def initialize_pool(self):
        """Initialize async connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.async_connection_string,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            logger.info("PostgreSQL connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def close_pool(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get async database connection"""
        if not self.pool:
            await self.initialize_pool()
            
        async with self.pool.acquire() as connection:
            yield connection
    
    # =================== Lead Management ===================
    
    async def get_leads(
        self, 
        domain: str = None,
        status: str = None,
        assigned_to: int = None,
        campaign_id: int = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get leads from database with filters"""
        try:
            query = """
                SELECT l.*, c.name as company_name, u.username as assigned_username
                FROM leads l
                LEFT JOIN companies c ON l.company_id = c.id
                LEFT JOIN users u ON l.assigned_to = u.id
                WHERE 1=1
            """
            params = []
            
            # Apply filters
            if domain:
                query += " AND l.domain = $" + str(len(params) + 1)
                params.append(domain)
            if status:
                query += " AND l.status = $" + str(len(params) + 1)
                params.append(status)
            if assigned_to:
                query += " AND l.assigned_to = $" + str(len(params) + 1)
                params.append(assigned_to)
            if campaign_id:
                query += " AND l.campaign_id = $" + str(len(params) + 1)
                params.append(campaign_id)
            
            # Add ordering and pagination
            query += " ORDER BY l.priority DESC, l.created_at DESC"
            query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
            params.extend([limit, offset])
            
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, *params)
                
                leads = []
                for row in rows:
                    lead = dict(row)
                    # Convert datetime objects to ISO strings
                    for key, value in lead.items():
                        if isinstance(value, datetime):
                            lead[key] = value.isoformat()
                    leads.append(lead)
                
                logger.info(f"Retrieved {len(leads)} leads")
                return leads
                
        except Exception as e:
            logger.error(f"Failed to get leads: {e}")
            return []
    
    async def get_lead_by_id(self, lead_id: int) -> Optional[Dict[str, Any]]:
        """Get specific lead by ID"""
        try:
            query = """
                SELECT l.*, c.name as company_name, u.username as assigned_username
                FROM leads l
                LEFT JOIN companies c ON l.company_id = c.id
                LEFT JOIN users u ON l.assigned_to = u.id
                WHERE l.id = $1
            """
            
            async with self.get_connection() as conn:
                row = await conn.fetchrow(query, lead_id)
                
                if row:
                    lead = dict(row)
                    # Convert datetime objects to ISO strings
                    for key, value in lead.items():
                        if isinstance(value, datetime):
                            lead[key] = value.isoformat()
                    return lead
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get lead {lead_id}: {e}")
            return None
    
    async def create_lead(self, lead_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new lead"""
        try:
            # Validate required fields
            validated_data = self.validate_lead_data(lead_data)
            
            query = """
                INSERT INTO leads (
                    name, email, phone, company_id, company, title, domain,
                    status, priority, source, assigned_to, campaign_id,
                    estimated_value, notes, tags, custom_fields
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                ) RETURNING *
            """
            
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    query,
                    validated_data["name"],
                    validated_data.get("email"),
                    validated_data["phone"],
                    validated_data.get("company_id"),
                    validated_data.get("company"),
                    validated_data.get("title"),
                    validated_data["domain"],
                    validated_data.get("status", "new"),
                    validated_data.get("priority", "medium"),
                    validated_data.get("source"),
                    validated_data.get("assigned_to"),
                    validated_data.get("campaign_id"),
                    validated_data.get("estimated_value"),
                    validated_data.get("notes"),
                    validated_data.get("tags", []),
                    json.dumps(validated_data.get("custom_fields", {}))
                )
                
                if row:
                    lead = dict(row)
                    for key, value in lead.items():
                        if isinstance(value, datetime):
                            lead[key] = value.isoformat()
                    
                    logger.info(f"Created lead: {lead['id']}")
                    return lead
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to create lead: {e}")
            return None
    
    async def update_lead(self, lead_id: int, updates: Dict[str, Any]) -> bool:
        """Update lead information"""
        try:
            # Build dynamic update query
            set_clauses = []
            params = []
            param_count = 1
            
            for key, value in updates.items():
                if key not in ['id', 'created_at']:  # Prevent updating these fields
                    set_clauses.append(f"{key} = ${param_count}")
                    params.append(value)
                    param_count += 1
            
            if not set_clauses:
                return False
            
            # Add updated_at
            set_clauses.append(f"updated_at = ${param_count}")
            params.append(datetime.utcnow())
            param_count += 1
            
            # Add lead_id for WHERE clause
            params.append(lead_id)
            
            query = f"""
                UPDATE leads 
                SET {', '.join(set_clauses)}
                WHERE id = ${param_count}
                RETURNING id
            """
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(query, *params)
                
                if result:
                    logger.info(f"Updated lead {lead_id}")
                    
                    # Log activity
                    await self.log_activity(
                        lead_id=lead_id,
                        activity_type="lead_updated",
                        details={"updated_fields": list(updates.keys())}
                    )
                    
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update lead {lead_id}: {e}")
            return False
    
    # =================== Call Logging ===================
    
    async def log_call(self, call_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Log call activity"""
        try:
            query = """
                INSERT INTO call_logs (
                    lead_id, campaign_id, user_id, ai_server_id, phone_number,
                    call_sid, direction, status, outcome, duration, recording_url,
                    ai_summary, ai_confidence, objections_handled, interest_level,
                    next_action, scheduled_follow_up, voice_engine, ai_model
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                ) RETURNING *
            """
            
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    query,
                    call_data.get("lead_id"),
                    call_data.get("campaign_id"),
                    call_data.get("user_id"),
                    call_data.get("ai_server_id"),
                    call_data["phone_number"],
                    call_data.get("call_sid"),
                    call_data.get("direction", "outbound"),
                    call_data.get("status"),
                    call_data.get("outcome"),
                    call_data.get("duration"),
                    call_data.get("recording_url"),
                    call_data.get("ai_summary"),
                    call_data.get("ai_confidence"),
                    call_data.get("objections_handled", []),
                    call_data.get("interest_level"),
                    call_data.get("next_action"),
                    call_data.get("scheduled_follow_up"),
                    call_data.get("voice_engine"),
                    call_data.get("ai_model")
                )
                
                if row:
                    call_log = dict(row)
                    # Convert datetime objects
                    for key, value in call_log.items():
                        if isinstance(value, datetime):
                            call_log[key] = value.isoformat()
                    
                    logger.info(f"Logged call: {call_log['id']}")
                    
                    # Update lead with last contact info
                    if call_data.get("lead_id"):
                        await self.update_lead(call_data["lead_id"], {
                            "last_contacted": datetime.utcnow(),
                            "call_count": await self.increment_call_count(call_data["lead_id"])
                        })
                    
                    return call_log
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to log call: {e}")
            return None
    
    async def increment_call_count(self, lead_id: int) -> int:
        """Increment and return call count for lead"""
        try:
            query = """
                UPDATE leads 
                SET call_count = COALESCE(call_count, 0) + 1 
                WHERE id = $1 
                RETURNING call_count
            """
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(query, lead_id)
                return result['call_count'] if result else 0
                
        except Exception as e:
            logger.error(f"Failed to increment call count: {e}")
            return 0
    
    # =================== User Management ===================
    
    async def get_users(self, domain: str = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get users from database"""
        try:
            query = "SELECT * FROM users WHERE 1=1"
            params = []
            
            if domain:
                query += " AND domain = $" + str(len(params) + 1)
                params.append(domain)
            
            if active_only:
                query += " AND is_active = true"
            
            # Hide system users from regular queries
            query += " AND is_system = false"
            query += " ORDER BY created_at DESC"
            
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, *params)
                
                users = []
                for row in rows:
                    user = dict(row)
                    # Remove password hash from response
                    user.pop('password_hash', None)
                    # Convert datetime objects
                    for key, value in user.items():
                        if isinstance(value, datetime):
                            user[key] = value.isoformat()
                    users.append(user)
                
                return users
                
        except Exception as e:
            logger.error(f"Failed to get users: {e}")
            return []
    
    async def authenticate_user(self, username: str, domain: str = None) -> Optional[Dict[str, Any]]:
        """Get user for authentication (including password hash)"""
        try:
            query = "SELECT * FROM users WHERE username = $1 AND is_active = true"
            params = [username]
            
            if domain:
                query += " AND domain = $2"
                params.append(domain)
            
            async with self.get_connection() as conn:
                row = await conn.fetchrow(query, *params)
                
                if row:
                    user = dict(row)
                    # Convert datetime objects
                    for key, value in user.items():
                        if isinstance(value, datetime):
                            user[key] = value.isoformat()
                    
                    # Update last login
                    await conn.execute(
                        "UPDATE users SET last_login = $1 WHERE id = $2",
                        datetime.utcnow(), user['id']
                    )
                    
                    return user
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to authenticate user: {e}")
            return None
    
    # =================== Campaign Management ===================
    
    async def get_campaigns(self, domain: str = None, user_id: int = None) -> List[Dict[str, Any]]:
        """Get campaigns from database"""
        try:
            query = """
                SELECT c.*, u.username as created_by_username,
                       COUNT(l.id) as lead_count
                FROM campaigns c
                LEFT JOIN users u ON c.created_by = u.id
                LEFT JOIN leads l ON c.id = l.campaign_id
                WHERE 1=1
            """
            params = []
            
            if domain:
                query += " AND c.domain = $" + str(len(params) + 1)
                params.append(domain)
            
            if user_id:
                query += " AND c.created_by = $" + str(len(params) + 1)
                params.append(user_id)
            
            query += " GROUP BY c.id, u.username ORDER BY c.created_at DESC"
            
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, *params)
                
                campaigns = []
                for row in rows:
                    campaign = dict(row)
                    # Convert datetime objects
                    for key, value in campaign.items():
                        if isinstance(value, datetime):
                            campaign[key] = value.isoformat()
                    campaigns.append(campaign)
                
                return campaigns
                
        except Exception as e:
            logger.error(f"Failed to get campaigns: {e}")
            return []
    
    # =================== Analytics ===================
    
    async def get_lead_stats(self, domain: str = None, date_range: int = 30) -> Dict[str, Any]:
        """Get lead statistics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=date_range)
            
            query = """
                SELECT 
                    COUNT(*) as total_leads,
                    COUNT(*) FILTER (WHERE status = 'new') as new_leads,
                    COUNT(*) FILTER (WHERE status = 'contacted') as contacted_leads,
                    COUNT(*) FILTER (WHERE status = 'qualified') as qualified_leads,
                    COUNT(*) FILTER (WHERE status = 'closed_won') as converted_leads,
                    COUNT(*) FILTER (WHERE status = 'closed_lost') as lost_leads
                FROM leads 
                WHERE created_at >= $1
            """
            params = [start_date]
            
            if domain:
                query += " AND domain = $2"
                params.append(domain)
            
            async with self.get_connection() as conn:
                row = await conn.fetchrow(query, *params)
                
                if row:
                    stats = dict(row)
                    
                    # Calculate conversion rate
                    if stats["contacted_leads"] > 0:
                        stats["conversion_rate"] = round((stats["converted_leads"] / stats["contacted_leads"]) * 100, 2)
                    else:
                        stats["conversion_rate"] = 0
                    
                    return stats
                
                return {"total_leads": 0, "conversion_rate": 0}
                
        except Exception as e:
            logger.error(f"Failed to get lead stats: {e}")
            return {"total_leads": 0, "conversion_rate": 0}
    
    async def get_call_analytics(self, campaign_id: int = None, days: int = 7) -> Dict[str, Any]:
        """Get call analytics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            query = """
                SELECT 
                    COUNT(*) as total_calls,
                    COUNT(*) FILTER (WHERE outcome = 'connected') as connected_calls,
                    COUNT(*) FILTER (WHERE outcome = 'voicemail') as voicemail_calls,
                    COUNT(*) FILTER (WHERE outcome = 'no_answer') as no_answer_calls,
                    COUNT(*) FILTER (WHERE outcome = 'failed') as failed_calls,
                    AVG(duration) FILTER (WHERE outcome = 'connected' AND duration IS NOT NULL) as avg_duration
                FROM call_logs 
                WHERE created_at >= $1
            """
            params = [start_date]
            
            if campaign_id:
                query += " AND campaign_id = $2"
                params.append(campaign_id)
            
            async with self.get_connection() as conn:
                row = await conn.fetchrow(query, *params)
                
                if row:
                    analytics = dict(row)
                    
                    # Convert avg_duration to float and round
                    if analytics.get("avg_duration"):
                        analytics["avg_duration"] = round(float(analytics["avg_duration"]), 1)
                    else:
                        analytics["avg_duration"] = 0
                    
                    # Calculate connection rate
                    if analytics["total_calls"] > 0:
                        analytics["connection_rate"] = round((analytics["connected_calls"] / analytics["total_calls"]) * 100, 2)
                    else:
                        analytics["connection_rate"] = 0
                    
                    return analytics
                
                return {"total_calls": 0, "connection_rate": 0}
                
        except Exception as e:
            logger.error(f"Failed to get call analytics: {e}")
            return {"total_calls": 0, "connection_rate": 0}
    
    # =================== Activity Logging ===================
    
    async def log_activity(
        self,
        lead_id: int = None,
        user_id: int = None,
        activity_type: str = None,
        subject: str = None,
        description: str = None,
        details: Dict[str, Any] = None
    ):
        """Log activity"""
        try:
            query = """
                INSERT INTO activities (
                    lead_id, user_id, activity_type, subject, description, completed_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """
            
            description_text = description or json.dumps(details) if details else None
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(
                    query,
                    lead_id,
                    user_id,
                    activity_type,
                    subject,
                    description_text,
                    datetime.utcnow()
                )
                
                if result:
                    logger.debug(f"Logged activity: {activity_type}")
                
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
    
    # =================== Bulk Operations ===================
    
    async def bulk_import_leads(self, leads_data: List[Dict[str, Any]], domain: str, campaign_id: int = None) -> int:
        """Bulk import leads"""
        try:
            imported_count = 0
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    for lead_data in leads_data:
                        try:
                            # Validate and prepare data
                            validated_data = self.validate_lead_data({
                                **lead_data,
                                "domain": domain,
                                "campaign_id": campaign_id,
                                "source": "bulk_import",
                                "status": "imported"
                            })
                            
                            # Insert lead
                            await conn.execute("""
                                INSERT INTO leads (
                                    name, email, phone, company, domain, status, 
                                    source, campaign_id, priority, notes
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            """,
                                validated_data["name"],
                                validated_data.get("email"),
                                validated_data["phone"],
                                validated_data.get("company"),
                                validated_data["domain"],
                                validated_data["status"],
                                validated_data["source"],
                                validated_data.get("campaign_id"),
                                validated_data.get("priority", "medium"),
                                validated_data.get("notes")
                            )
                            
                            imported_count += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to import lead {lead_data.get('name', 'Unknown')}: {e}")
                            continue
                    
            logger.info(f"Bulk imported {imported_count} leads")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to bulk import leads: {e}")
            return 0
    
    # =================== Utility Functions ===================
    
    def validate_lead_data(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean lead data"""
        validated = {}
        
        # Required fields
        required_fields = ["name", "phone", "domain"]
        for field in required_fields:
            if field not in lead_data or not lead_data[field]:
                raise ValueError(f"Missing required field: {field}")
            validated[field] = lead_data[field]
        
        # Optional fields with defaults
        validated["email"] = lead_data.get("email", "")
        validated["company"] = lead_data.get("company", "")
        validated["title"] = lead_data.get("title", "")
        validated["status"] = lead_data.get("status", "new")
        validated["priority"] = lead_data.get("priority", "medium")
        validated["source"] = lead_data.get("source", "unknown")
        validated["notes"] = lead_data.get("notes", "")
        
        # Clean phone number
        phone = validated["phone"].replace("-", "").replace("(", "").replace(")", "").replace(" ", "")
        if not phone.startswith("+"):
            if len(phone) == 10:
                phone = "+1" + phone
            elif len(phone) == 11 and phone.startswith("1"):
                phone = "+" + phone
        
        validated["phone"] = phone
        
        # Validate email format
        email = validated["email"]
        if email and "@" not in email:
            raise ValueError("Invalid email format")
        
        return validated
    
    async def execute_raw_query(self, query: str, params: List = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query"""
        try:
            async with self.get_connection() as conn:
                if params:
                    rows = await conn.fetch(query, *params)
                else:
                    rows = await conn.fetch(query)
                
                results = []
                for row in rows:
                    result = dict(row)
                    # Convert datetime objects
                    for key, value in result.items():
                        if isinstance(value, datetime):
                            result[key] = value.isoformat()
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []

# Global PostgreSQL instance
postgresql_crm = PostgreSQLIntegration()

# FastAPI dependency
async def get_postgresql_crm() -> PostgreSQLIntegration:
    """FastAPI dependency for PostgreSQL CRM"""
    return postgresql_crm

# Context manager for database operations
class PostgreSQLCRMContext:
    def __init__(self):
        self.crm = postgresql_crm
    
    async def __aenter__(self):
        return self.crm
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass