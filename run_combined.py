"""
Combined runner for Streamlit app and FastAPI endpoints
Serves both the web interface and API on different ports
"""

import subprocess
import threading
import time
import signal
import sys
import os
from contextlib import contextmanager

def run_streamlit():
    """Run Streamlit app on port 5000"""
    cmd = ["streamlit", "run", "app.py", "--server.port", "5000", "--server.address", "0.0.0.0"]
    return subprocess.Popen(cmd)

def run_fastapi():
    """Run FastAPI app on port 8000"""
    cmd = ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    return subprocess.Popen(cmd)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\nReceived signal {signum}. Shutting down...")
    sys.exit(0)

def main():
    """Main function to run both applications"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Starting Google Earth Engine Dashboard...")
    print("📱 Streamlit UI will be available on port 5000")
    print("🔗 FastAPI endpoints will be available on port 8000")
    print("📖 API documentation at http://localhost:8000/docs")
    
    # Start both applications
    streamlit_process = None
    fastapi_process = None
    
    try:
        # Start FastAPI
        print("Starting FastAPI server...")
        fastapi_process = run_fastapi()
        time.sleep(2)
        
        # Start Streamlit
        print("Starting Streamlit app...")
        streamlit_process = run_streamlit()
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if fastapi_process.poll() is not None:
                print("FastAPI process died, restarting...")
                fastapi_process = run_fastapi()
                
            if streamlit_process.poll() is not None:
                print("Streamlit process died, restarting...")
                streamlit_process = run_streamlit()
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up processes
        if fastapi_process:
            fastapi_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()
        
        # Wait for processes to terminate
        if fastapi_process:
            fastapi_process.wait()
        if streamlit_process:
            streamlit_process.wait()

if __name__ == "__main__":
    main()