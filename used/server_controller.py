from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import subprocess
import signal
import os
import sys
import time

app = FastAPI(
    title="Server Manager API",
    description="API to manage Python server scripts from the frontend (e.g. React on Raspberry Pi)",
    version="1.0.0"
)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global Process State ===
process = None
current_script = None

class ServerRequest(BaseModel):
    script_name: str  # e.g., "server_mbs.py"

@app.post("/run")
def run_server(req: ServerRequest):
    global process, current_script

    if process and process.poll() is None:
        stop_server()

    try:
        script = req.script_name
        args = [sys.executable, script]

        process = subprocess.Popen(
            args,
            preexec_fn=os.setsid if os.name != 'nt' else None,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        current_script = script

        return JSONResponse(
            status_code=200,
            content={
                "status": "running",
                "script": script,
                "pid": process.pid,
                "message": f"{script} is now running (PID: {process.pid})"
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/stop")
def stop_server():
    global process, current_script

    if process and process.poll() is None:
        try:
            if os.name == 'nt':
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)

            process.wait(timeout=5)
        except Exception as e:
            return {"status": "error", "message": f"Failed to stop: {e}"}
        finally:
            process = None
            current_script = None

        return {"status": "stopped", "message": "Server stopped successfully."}

    return {"status": "stopped", "message": "No server is currently running."}

@app.post("/restart")
def restart_server(req: ServerRequest):
    stop_result = stop_server()

    if stop_result.get("status") != "stopped":
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Failed to stop server before restarting."}
        )

    # Delay to allow OS cleanup
    time.sleep(1.5)

    return run_server(req)

@app.get("/status")
def get_status():
    if process and process.poll() is None:
        return {
            "status": "running",
            "script": current_script,
            "pid": process.pid,
            "message": f"{current_script} is running (PID: {process.pid})"
        }

    return {
        "status": "stopped",
        "script": None,
        "pid": None,
        "message": "No server is currently running."
    }

# Graceful shutdown on Ctrl+C
def handle_sigint(signal_received, frame):
    print("SIGINT received. Stopping server...")
    response = stop_server()
    print("Stop result:", response)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)
