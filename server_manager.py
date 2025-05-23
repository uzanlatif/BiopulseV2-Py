import tkinter as tk
from tkinter import messagebox
import subprocess
import signal

class ServerManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Server Manager")
        
        self.selected_server = tk.StringVar(value="server_mbs.py")
        self.process = None

        # Radio buttons untuk memilih server
        tk.Radiobutton(root, text="Server MBS", variable=self.selected_server, value="server_mbs.py").pack(anchor='w')
        tk.Radiobutton(root, text="Server ECG", variable=self.selected_server, value="server_ecg.py").pack(anchor='w')
        tk.Radiobutton(root, text="Server EEG", variable=self.selected_server, value="server_eeg.py").pack(anchor='w')

        # Tombol untuk menjalankan server
        tk.Button(root, text="Run Server", command=self.run_selected_server).pack(pady=10)

        # Tombol untuk stop server
        tk.Button(root, text="Stop Server", command=self.stop_server).pack(pady=5)

    def run_selected_server(self):
        # Jika ada proses server yang berjalan, hentikan dulu
        if self.process and self.process.poll() is None:
            self.stop_server()

        server_script = self.selected_server.get()
        try:
            # Jalankan server script baru
            self.process = subprocess.Popen(["python", server_script])
            messagebox.showinfo("Info", f"{server_script} is running.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run {server_script}:\n{e}")

    def stop_server(self):
        if self.process and self.process.poll() is None:
            # Menghentikan proses server yang berjalan
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            messagebox.showinfo("Info", "Server stopped.")
            self.process = None
        else:
            messagebox.showinfo("Info", "No server is running.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ServerManagerApp(root)
    root.mainloop()
