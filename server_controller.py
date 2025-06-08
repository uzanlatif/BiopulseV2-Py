from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import signal
import os
import sys
from fastapi.middleware.cors import CORSMiddleware
import logging

# Logger setup (optional but useful for debugging)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_manager")

# FastAPI instance
app = FastAPI(
    title="Server Manager API",
    description="API to manage server scripts from frontend",
    version="1.0.0"
)

# CORS setup (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to track running process
process = None
current_script = None

# Pydantic model for POST body
class ServerRequest(BaseModel):
    script_name: str  # e.g. "server_mbs_ssl.py"

@app.post("/run")
def run_server(req: ServerRequest):
    """
    Jalankan script server baru, hentikan yang lama jika masih berjalan.
    """
    global process, current_script

    # Stop old process if running
    if process and process.poll() is None:
        stop_server()

    try:
        script = req.script_name
        logger.info(f"Starting new server script: {script}")
        process = subprocess.Popen(
            [sys.executable, script],
            preexec_fn=os.setsid if os.name != 'nt' else None,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        current_script = script
        return {
            "status": "running",
            "script": script,
            "pid": process.pid
        }
    except Exception as e:
        logger.error(f"Failed to start script: {e}")
        return {"error": str(e)}

@app.post("/stop")
def stop_server():
    """
    Hentikan script server yang sedang berjalan.
    """
    global process, current_script

    if process and process.poll() is None:
        try:
            logger.info("Stopping running server script...")
            if os.name == 'nt':
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error while stopping script: {e}")
            return {"error": str(e)}
        finally:
            process = None
            current_script = None
        return {"status": "stopped"}
    return {"status": "no server running"}

@app.get("/status")
def get_status():
    """
    Periksa status server saat ini.
    """
    if process and process.poll() is None:
        return {
            "status": "running",
            "script": current_script,
            "pid": process.pid
        }
    return {"status": "stopped"}

@app.on_event("shutdown")
def shutdown_event():
    """
    Otomatis dipanggil saat Ctrl+C ditekan di terminal Raspberry Pi.
    """
    logger.info("Shutdown triggered (Ctrl+C). Cleaning up running process...")
    stop_server()
