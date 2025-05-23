import tkinter as tk
from tkinter import messagebox
import subprocess
import signal
import os
import sys

class ServerManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Server Manager")
        
        self.selected_server = tk.StringVar(value="server_mbs.py")
        self.process = None

        # GUI: Radio buttons for server selection
        tk.Radiobutton(root, text="Server MBS", variable=self.selected_server, value="server_mbs.py").pack(anchor='w')
        tk.Radiobutton(root, text="Server ECG", variable=self.selected_server, value="server_ecg.py").pack(anchor='w')
        tk.Radiobutton(root, text="Server EEG", variable=self.selected_server, value="server_eeg.py").pack(anchor='w')

        # GUI: Run & Stop buttons
        tk.Button(root, text="Run Server", command=self.run_selected_server).pack(pady=10)
        tk.Button(root, text="Stop Server", command=self.stop_server).pack(pady=5)

    def run_selected_server(self):
        self.stop_server()  # Stop any existing server

        script = self.selected_server.get()
        try:
            # Start subprocess in a new process group
            self.process = subprocess.Popen(
                [sys.executable, script],
                preexec_fn=os.setsid if os.name != 'nt' else None,  # Unix
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0  # Windows
            )
            messagebox.showinfo("Info", f"{script} is running.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server:\n{e}")

    def stop_server(self):
        if self.process and self.process.poll() is None:
            try:
                if os.name == 'nt':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)  # Windows
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)  # Unix

                self.process.wait(timeout=5)
                messagebox.showinfo("Info", "Server stopped.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop server:\n{e}")
            finally:
                self.process = None
        else:
            self.process = None  # Clear handle just in case

if __name__ == "__main__":
    root = tk.Tk()
    app = ServerManagerApp(root)
    root.mainloop()
