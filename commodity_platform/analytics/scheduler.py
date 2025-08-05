#!/usr/bin/env python3
"""
Scheduler Module for PySpark ETL Jobs
Uses APScheduler to run ETL jobs on a schedule.
"""

import logging
import sys
import os
from datetime import datetime, time
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
import atexit

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.spark_etl import CommoditySparkETL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ETLScheduler:
    def __init__(self):
        """Initialize the ETL scheduler"""
        self.scheduler = None
        self.setup_scheduler()
    
    def setup_scheduler(self):
        """Setup APScheduler with job store and executors"""
        try:
            # Configure job store (SQLite for persistence)
            jobstores = {
                'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
            }
            
            # Configure executors
            executors = {
                'default': ThreadPoolExecutor(max_workers=2),
            }
            
            # Job defaults
            job_defaults = {
                'coalesce': False,
                'max_instances': 1,
                'misfire_grace_time': 3600  # 1 hour grace period
            }
            
            # Create scheduler
            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone='UTC'
            )
            
            logger.info("ETL Scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            raise
    
    def run_daily_etl_job(self):
        """Run the daily ETL job"""
        job_start = datetime.now()
        logger.info(f"Starting scheduled ETL job at {job_start}")
        
        try:
            # Initialize ETL
            etl = CommoditySparkETL(app_name="ScheduledETL")
            
            # Run ETL pipeline with sample data
            success = etl.run_etl_pipeline(
                use_sample_data=True, 
                num_records=1000000
            )
            
            if success:
                logger.info("✅ Scheduled ETL job completed successfully")
                
                # Optional: Send notification or update status
                self._notify_job_completion(True, job_start)
                
            else:
                logger.error("❌ Scheduled ETL job failed")
                self._notify_job_completion(False, job_start)
            
        except Exception as e:
            logger.error(f"ETL job failed with error: {e}")
            self._notify_job_completion(False, job_start, str(e))
        
        finally:
            job_end = datetime.now()
            duration = (job_end - job_start).total_seconds()
            logger.info(f"ETL job finished at {job_end}, duration: {duration:.2f} seconds")
    
    def _notify_job_completion(self, success: bool, start_time: datetime, error: str = None):
        """Notify about job completion (could be extended to send emails, webhooks, etc.)"""
        status = "SUCCESS" if success else "FAILED"
        duration = (datetime.now() - start_time).total_seconds()
        
        notification = {
            "job": "daily_etl",
            "status": status,
            "start_time": start_time.isoformat(),
            "duration_seconds": duration,
            "error": error
        }
        
        # Save notification to file (could be extended to database, webhook, etc.)
        try:
            import json
            log_file = "/workspace/commodity_platform/data/analytics/job_log.json"
            
            # Load existing logs
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Add new log
            logs.append(notification)
            
            # Keep only last 100 logs
            logs = logs[-100:]
            
            # Save logs
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"Job notification saved: {status}")
            
        except Exception as e:
            logger.warning(f"Could not save job notification: {e}")
    
    def schedule_daily_job(self, hour: int = 2, minute: int = 0):
        """Schedule daily ETL job"""
        try:
            # Remove existing job if it exists
            if self.scheduler.get_job('daily_etl'):
                self.scheduler.remove_job('daily_etl')
            
            # Add daily job
            self.scheduler.add_job(
                func=self.run_daily_etl_job,
                trigger=CronTrigger(hour=hour, minute=minute),
                id='daily_etl',
                name='Daily Commodity ETL Job',
                replace_existing=True
            )
            
            logger.info(f"Daily ETL job scheduled for {hour:02d}:{minute:02d} UTC")
            
        except Exception as e:
            logger.error(f"Failed to schedule daily job: {e}")
            raise
    
    def schedule_weekly_job(self, day_of_week: int = 0, hour: int = 1):
        """Schedule weekly comprehensive ETL job (optional)"""
        try:
            # Remove existing job if it exists
            if self.scheduler.get_job('weekly_etl'):
                self.scheduler.remove_job('weekly_etl')
            
            # Add weekly job with more data
            self.scheduler.add_job(
                func=lambda: self._run_comprehensive_etl(),
                trigger=CronTrigger(day_of_week=day_of_week, hour=hour),
                id='weekly_etl',
                name='Weekly Comprehensive ETL Job',
                replace_existing=True
            )
            
            logger.info(f"Weekly ETL job scheduled for day {day_of_week} at {hour:02d}:00 UTC")
            
        except Exception as e:
            logger.error(f"Failed to schedule weekly job: {e}")
    
    def _run_comprehensive_etl(self):
        """Run comprehensive ETL with more data"""
        logger.info("Starting weekly comprehensive ETL job")
        
        try:
            etl = CommoditySparkETL(app_name="WeeklyETL")
            
            # Run with larger dataset
            success = etl.run_etl_pipeline(
                use_sample_data=True, 
                num_records=5000000  # 5M records for weekly job
            )
            
            if success:
                logger.info("✅ Weekly comprehensive ETL completed")
            else:
                logger.error("❌ Weekly comprehensive ETL failed")
                
        except Exception as e:
            logger.error(f"Weekly ETL failed: {e}")
    
    def run_manual_job(self, num_records: int = 1000000):
        """Run ETL job manually (for testing)"""
        logger.info("Running manual ETL job")
        self.run_daily_etl_job()
    
    def start(self):
        """Start the scheduler"""
        try:
            if not self.scheduler.running:
                self.scheduler.start()
                logger.info("ETL Scheduler started")
                
                # Register shutdown handler
                atexit.register(lambda: self.stop())
            else:
                logger.warning("Scheduler is already running")
                
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def stop(self):
        """Stop the scheduler"""
        try:
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown()
                logger.info("ETL Scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    def get_job_status(self) -> dict:
        """Get status of scheduled jobs"""
        try:
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                    'trigger': str(job.trigger)
                })
            
            return {
                'scheduler_running': self.scheduler.running,
                'jobs': jobs,
                'total_jobs': len(jobs)
            }
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {'error': str(e)}
    
    def get_job_history(self) -> list:
        """Get job execution history"""
        try:
            log_file = "/workspace/commodity_platform/data/analytics/job_log.json"
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    return json.load(f)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error reading job history: {e}")
            return []

# Global scheduler instance
_scheduler_instance = None

def get_scheduler() -> ETLScheduler:
    """Get global scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = ETLScheduler()
    return _scheduler_instance

def start_scheduler():
    """Start the global scheduler"""
    scheduler = get_scheduler()
    
    # Schedule daily job at 2 AM UTC
    scheduler.schedule_daily_job(hour=2, minute=0)
    
    # Schedule weekly job on Sunday at 1 AM UTC
    scheduler.schedule_weekly_job(day_of_week=6, hour=1)
    
    # Start scheduler
    scheduler.start()
    
    logger.info("ETL Scheduler service started")
    return scheduler

def main():
    """Main function for standalone scheduler execution"""
    print("🕒 PySpark ETL Scheduler")
    print("=" * 50)
    
    try:
        # Start scheduler
        scheduler = start_scheduler()
        
        # Show status
        status = scheduler.get_job_status()
        print(f"📅 Scheduler Status: {'Running' if status['scheduler_running'] else 'Stopped'}")
        print(f"📋 Total Jobs: {status['total_jobs']}")
        
        for job in status['jobs']:
            print(f"   • {job['name']}: Next run at {job['next_run']}")
        
        print("\n💡 Scheduler is running in background...")
        print("💡 Press Ctrl+C to stop")
        
        # Keep the script running
        import time
        while True:
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping scheduler...")
        if 'scheduler' in locals():
            scheduler.stop()
        print("✅ Scheduler stopped")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()