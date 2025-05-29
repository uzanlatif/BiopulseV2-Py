import asyncio
import websockets
import json
import time
import signal
import sys
import ssl
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

# Global board instance
board = None
board_initialized = False

# Signal handler
def signal_handler(sig, frame):
    print("\n🛑 Signal received, cleaning up...")
    cleanup()
    sys.exit(0)

def cleanup():
    global board, board_initialized
    if board and board_initialized:
        try:
            board.stop_stream()
        except BrainFlowError as e:
            print("⚠️ stop_stream error:", e)
        try:
            board.release_session()
        except BrainFlowError as e:
            print("⚠️ release_session error:", e)
    board_initialized = False
    print("✅ Cleaned up")

# Register signal handler
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# EEG handler
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        sampling_rate = board.get_sampling_rate(board_id)
        interval = 1.0 / sampling_rate

        while True:
            raw_data = board.get_current_board_data(50)
            print(f"[DEBUG] raw_data shape: {raw_data.shape}")

            if raw_data.shape[1] == 0:
                print("[⚠️ WARNING] No data received from board yet.")
                await asyncio.sleep(0.5)
                continue

            sensor_data = {}
            timestamp_now = time.time()

            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                samples = raw_data[ch]

                # 🧪 Log nilai min/max
                print(f"[DEBUG] {label} ➜ Min: {min(samples):.2f}, Max: {max(samples):.2f}")

                sensor_data[label] = [
                    {
                        "y": float(val),
                        "__timestamp__": timestamp_now - (len(samples) - i - 1) * interval
                    }
                    for i, val in enumerate(samples)
                ]

            await websocket.send(json.dumps(sensor_data))
            await asyncio.sleep(0.3)
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# BrainFlow config
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # Ganti jika port berbeda
board_id = BoardIds.CYTON_DAISY_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)

channel_names = {
    1: "ECG", 2: "PPG", 3: "PCG", 4: "EMG1", 5: "EMG2",
    6: "MYOMETER", 7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP", 10: "OXYGEN",
    11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13", 14: "EEG CH14",
    15: "EEG CH15", 16: "EEG CH16", 17: "EEG CH17", 18: "EEG CH18",
    19: "EEG CH19", 20: "EEG CH20", 21: "EEG CH21", 22: "EEG CH22",
    23: "EEG CH23", 24: "EEG CH24"
}

async def main():
    global board, board_initialized
    board = BoardShim(board_id, params)

    try:
        print("🔄 Preparing BrainFlow session...")
        board.prepare_session()

        # ✅ Konfigurasikan gain maksimum (24) ke semua 16 channel EEG
        config_gain_24 = (
            'x1240000Xx2240000Xx3240000Xx4240000Xx5240000Xx6240000Xx7240000Xx8240000X' +
            'xQ240000XxW240000XxE240000XxR240000XxT240000XxY240000XxU240000XxI240000X'
        )
        board.config_board(config_gain_24)
        time.sleep(0.5)  # beri waktu apply

        board.start_stream()
        time.sleep(1)  # tunggu streaming
        board_initialized = True
        print("✅ MBS streaming started")

        # 🔎 Cek apakah data tersedia
        print("🔎 Checking data availability...")
        time.sleep(3)
        data = board.get_board_data()
        print(f"[DEBUG] Initial board data shape: {data.shape}")

        ip = '0.0.0.0'
        port = 5555

        current_dir = os.path.dirname(os.path.abspath(__file__))
        cert_path = os.path.join(current_dir, "cert.pem")
        key_path = os.path.join(current_dir, "key.pem")

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)

        async with websockets.serve(
            eeg_handler, ip, port, ssl=ssl_context
        ):
            print(f"🔒 Secure WebSocket (WSS) running at wss://<raspi-ip>:{port}")
            await asyncio.Future()  # run forever
    except BrainFlowError as e:
        print("🚨 BrainFlow error:", e)
    except Exception as e:
        print("🚨 Unexpected error:", e)
    finally:
        cleanup()

if __name__ == '__main__':
    asyncio.run(main())
