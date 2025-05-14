import asyncio
import json
import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from websockets.legacy.server import serve  # ✅ PENTING: gunakan legacy API

# Konfigurasi BrainFlow untuk Cyton + Daisy
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # Ubah jika port berbeda

board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)

# Dapatkan channel EEG
eeg_channels = BoardShim.get_eeg_channels(board_id)

# Label channel (opsional)
channel_names = {
    1: "ECG", 2: "PPG", 3: "PCG", 4: "EMG1", 5: "EMG2", 6: "MYOMETER",
    7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP", 10: "OXYGEN",
    11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13",
    14: "EEG CH14", 15: "EEG CH15", 16: "EEG CH16"
}

# Handler WebSocket: wajib dua parameter untuk legacy API
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        while True:
            data = board.get_current_board_data(50)  # Ambil 50 sample terbaru
            eeg_data = {}
            timestamp = int(time.time() * 1000)  # Timestamp dalam milidetik (UNIX epoch)
            
            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                eeg_data[label] = {
                    'values': data[ch].tolist(),
                    'timestamp': timestamp  # Menambahkan timestamp
                }

            # Kirim data dengan timestamp
            await websocket.send(json.dumps(eeg_data))
            await asyncio.sleep(0.05)  # Kirim data setiap 50ms (~20Hz)
    except Exception as e:
        print(f"❌ Client disconnected: {e}")

# Fungsi utama menjalankan server
async def main():
    ip = '192.168.45.249'  # Ganti dengan IP Raspberry Pi kamu
    port = 8765

    print("🔄 Preparing board session...")
    board.prepare_session()
    board.start_stream()
    print(f"✅ EEG data streaming from board")

    async with serve(eeg_handler, ip, port):
        print(f"✅ WebSocket Server running at ws://{ip}:{port}")
        await asyncio.Future()  # Run forever

# Jalankan script
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    finally:
        print("🧹 Releasing BrainFlow session...")
        board.stop_stream()
        board.release_session()
        print("✅ Cleanup done")
