import asyncio
import websockets
import json
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# 🔧 Konfigurasi BrainFlow
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # Ganti sesuai port Raspberry Pi

board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)

# 🧠 Daftar channel EEG yang tersedia
eeg_channels = BoardShim.get_eeg_channels(board_id)

# 🏷️ Label channel (bisa disesuaikan)
channel_names = {
    1: "ECG", 
    2: "PPG", 
    3: "PCG", 
    4: "EMG1", 
    5: "EMG2", 
    6: "MYOMETER",
    7: "SPIRO", 
    8: "TEMPERATURE", 
    9: "NIBP", 
    10: "OXYGEN",
    11: "EEG CH11", 
    12: "EEG CH12", 
    13: "EEG CH13",
    14: "EEG CH14", 
    15: "EEG CH15", 
    16: "EEG CH16"
}

# 🌐 Handler WebSocket
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        while True:
            # Ambil 50 sample terakhir dari board
            data = board.get_current_board_data(50)
            n_samples = data.shape[1]
            timestamps = data[-1]  # channel terakhir = timestamp

            samples = []
            for i in range(n_samples):
                sample = {}
                for ch in eeg_channels:
                    label = channel_names.get(ch, f"CH{ch}")
                    sample[label] = float(data[ch][i])
                sample["__timestamp__"] = float(timestamps[i])  # UNIX timestamp (detik)
                samples.append(sample)

            await websocket.send(json.dumps(samples))
            await asyncio.sleep(0.3)  # frekuensi ~3Hz
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")

# 🚀 Fungsi utama menjalankan server
async def main():
    ip = '192.168.45.249'  # Ganti dengan IP Raspberry Pi
    port = 8765

    print("🔄 Preparing board session...")
    board.prepare_session()
    board.start_stream()
    print("✅ Streaming started")

    async with websockets.serve(eeg_handler, ip, port):
        print(f"✅ WebSocket Server running at ws://{ip}:{port}")
        await asyncio.Future()  # run forever

# 🔧 Run
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        print("🧹 Cleaning up session...")
        board.stop_stream()
        board.release_session()
        print("✅ Shutdown complete")
