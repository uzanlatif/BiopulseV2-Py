import asyncio
import websockets
import json
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Konfigurasi BrainFlow
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # Ganti dengan port USB Raspberry Pi

# Inisialisasi board
board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)

# Channel EEG dan label sensor
eeg_channels = BoardShim.get_eeg_channels(board_id)
channel_names = {
    1: "EEG_1", 
    2: "EEG_2", 
    3: "EEG_3", 
    4: "EEG_4", 
    5: "EEG_5", 
    6: "EEG_6",
    7: "EEG_7", 
    8: "EEG_8", 
    9: "EEG_9", 
    10: "EEG_10",
    11: "EEG_11", 
    12: "EEG_12", 
    13: "EEG_13", 
    14: "EEG_14",
    15: "EEG_15", 
    16: "EEG_16"
}

# WebSocket handler
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        sampling_rate = board.get_sampling_rate(board_id)  # e.g. 250
        interval = 1.0 / sampling_rate

        while True:
            raw_data = board.get_current_board_data(50)
            sensor_data = {}

            timestamp_now = time.time()  # UNIX time in seconds

            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                samples = raw_data[ch]
                sensor_data[label] = [
                    {
                        "y": float(val),
                        "__timestamp__": timestamp_now - (len(samples) - i - 1) * interval
                    }
                    for i, val in enumerate(samples)
                ]

            await websocket.send(json.dumps(sensor_data))
            await asyncio.sleep(0.3)  # Send data ~3.3Hz
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Server error:", e)

# Fungsi utama
async def main():
    ip = '172.30.81.62'  # Raspi IP Address
    port = 7777

    print("🔄 Preparing board session...")
    board.prepare_session()
    board.start_stream()
    print(f"✅ EEG data streaming from board")

    async with websockets.serve(eeg_handler, ip, port):
        print(f"🌐 WebSocket Server running at ws://{ip}:{port}")
        await asyncio.Future()  # keep alive

# Eksekusi
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
