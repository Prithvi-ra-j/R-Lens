#!/usr/bin/env python3
"""
Simple runner script to start the Commodity Price Monitoring Platform.
This script helps users quickly start both the API and dashboard.
"""

import subprocess
import sys
import time
import threading
from pathlib import Path

def run_api():
    """Run the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
        subprocess.run(cmd, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("📊 Starting Streamlit dashboard...")
    time.sleep(3)  # Wait a bit for API to start
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", "8501"]
        subprocess.run(cmd, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")

def main():
    """Main function to start both services"""
    print("=" * 60)
    print("🚀 Commodity Price Monitoring Platform")
    print("=" * 60)
    print("📈 API will start on: http://localhost:8000")
    print("📊 Dashboard will start on: http://localhost:8501")
    print("📚 API Docs available at: http://localhost:8000/docs")
    print("=" * 60)
    
    try:
        # Start API in a separate thread
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Start dashboard in main thread
        run_dashboard()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down platform...")
        sys.exit(0)

if __name__ == "__main__":
    main()