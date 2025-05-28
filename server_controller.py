from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import signal
import os
import sys
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Server Manager API",
    description="API to manage server scripts from frontend",
    version="1.0.0"
)

# CORS: allow all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti ke ['http://your-frontend.com'] jika perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
process = None
current_script = None

class ServerRequest(BaseModel):
    script_name: str  # e.g. "server_mbs_ssl.py"

@app.post("/run")
def run_server(req: ServerRequest):
    """
    Jalankan script server baru, hentikan yang lama jika masih jalan.
    """
    global process, current_script

    # Hentikan jika ada server lain sedang jalan
    if process and process.poll() is None:
        stop_server()

    try:
        script = req.script_name
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
        return {"error": str(e)}

@app.post("/stop")
def stop_server():
    """
    Hentikan script server yang sedang berjalan.
    """
    global process, current_script

    if process and process.poll() is None:
        try:
            if os.name == 'nt':
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except Exception as e:
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
