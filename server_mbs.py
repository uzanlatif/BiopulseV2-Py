import asyncio
import websockets
import json
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Konfigurasi BrainFlow
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # Ganti sesuai dengan port USB OpenBCI di Raspberry Pi

# Inisialisasi board
board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)

# Daftar channel EEG yang tersedia
eeg_channels = BoardShim.get_eeg_channels(board_id)

# Label channel custom (bisa disesuaikan)
channel_names = {
    1: "ECG", 2: "PPG", 3: "PCG", 4: "EMG1", 5: "EMG2", 6: "MYOMETER",
    7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP", 10: "OXYGEN",
    11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13",
    14: "EEG CH14", 15: "EEG CH15", 16: "EEG CH16"
}

# Fungsi handler WebSocket (wajib 2 parameter)
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        while True:
            # Ambil 50 sample data terbaru
            data = board.get_current_board_data(50)
            eeg_data = {}
            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                eeg_data[label] = data[ch].tolist()
            await websocket.send(json.dumps(eeg_data))
            await asyncio.sleep(0.05)  # 20 Hz update rate
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")

# Fungsi utama menjalankan WebSocket server
async def main():
    ip = '192.168.45.249'  # Ganti dengan IP Raspberry Pi kamu
    port = 8765

    # Persiapkan koneksi board
    print("🔄 Preparing board session...")
    board.prepare_session()
    board.start_stream()
    print(f"✅ EEG data streaming from board")

    # Jalankan WebSocket server
    async with websockets.serve(eeg_handler, ip, port):
        print(f"✅ WebSocket Server running at ws://{ip}:{port}")
        await asyncio.Future()  # Keep running forever

# Jalankan script
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
    finally:
        print("🧹 Cleaning up BrainFlow session...")
        board.stop_stream()
        board.release_session()
        print("✅ Done")
