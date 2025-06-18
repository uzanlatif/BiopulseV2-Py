from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import signal
import os
import sys
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Server Manager API",
    description="API to manage Python server scripts from the frontend (e.g. React on Raspberry Pi)",
    version="1.0.0"
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti ke ['http://localhost:3000'] jika frontend React di-local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global State ===
process = None
current_script = None

# === Request Model ===
class ServerRequest(BaseModel):
    script_name: str  # e.g. "server_mbs_ssl.py"


@app.post("/run")
def run_server(req: ServerRequest):
    """
    Jalankan script baru dan hentikan script lama jika masih berjalan.
    """
    global process, current_script

    # Hentikan proses sebelumnya jika masih aktif
    if process and process.poll() is None:
        stop_server()

    try:
        script = req.script_name
        process_args = [sys.executable, script]
        process = subprocess.Popen(
            process_args,
            preexec_fn=os.setsid if os.name != 'nt' else None,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        current_script = script

        return {
            "status": "running",
            "script": script,
            "pid": process.pid,
            "message": f"{script} is now running (PID: {process.pid})"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/stop")
def stop_server():
    """
    Hentikan script yang sedang berjalan.
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
            return {"status": "error", "message": str(e)}
        finally:
            process = None
            current_script = None
        return {"status": "stopped", "message": "Server stopped successfully."}

    return {"status": "stopped", "message": "No server is currently running."}


@app.post("/restart")
def restart_server(req: ServerRequest):
    """
    Restart script server: stop and start again.
    """
    global process, current_script

    try:
        # Stop jika sedang berjalan
        if process and process.poll() is None:
            stop_server()

        # Mulai ulang
        script = req.script_name
        process_args = [sys.executable, script]
        process = subprocess.Popen(
            process_args,
            preexec_fn=os.setsid if os.name != 'nt' else None,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        current_script = script

        return {
            "status": "restarted",
            "script": script,
            "pid": process.pid,
            "message": f"{script} restarted (PID: {process.pid})"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/status")
def get_status():
    """
    Ambil status server saat ini.
    """
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


# === Signal Handler (Ctrl+C)
def handle_sigint(signal_received, frame):
    print("SIGINT diterima. Menghentikan server jika aktif...")
    response = stop_server()
    print("Status penghentian:", response)
    print("Shutdown aplikasi FastAPI.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)
