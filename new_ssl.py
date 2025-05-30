import asyncio
import websockets
import json
import time
import signal
import sys
import ssl
import os
import numpy as np
from scipy.signal import iirnotch, filtfilt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

# === Global setup ===
board = None
board_initialized = False
use_notch_filter = True  # Set True to match GUI filtering

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

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# === Filter setup ===
def notch_filter(data, freq=60, fs=250, Q=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)

# === Handler ===
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        sampling_rate = board.get_sampling_rate(board_id)
        interval = 1.0 / sampling_rate

        while True:
            raw_data = board.get_board_data(50)  # shape: [channels, samples]
            if raw_data.shape[1] == 0:
                print("[⚠️ WARNING] No data received from board yet.")
                await asyncio.sleep(0.5)
                continue

            sensor_data = {}
            timestamp_now = time.time()

            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                samples = raw_data[ch]

                # Convert ADC raw to voltage (Cyton ADC = 24-bit, scale ~4.5V)
                scale = 4.5 / (2**23 - 1)
                volts = samples * scale  # now in Volts

                if use_notch_filter:
                    volts = notch_filter(volts, freq=60, fs=sampling_rate)

                # Match GUI value (float)
                sensor_data[label] = [
                    {
                        "y": float(round(val * 1000, 4)),  # Convert to mV, rounded
                        "__timestamp__": timestamp_now - (len(volts) - i - 1) * interval
                    }
                    for i, val in enumerate(volts)
                ]

            await websocket.send(json.dumps(sensor_data))
            await asyncio.sleep(0.3)
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# === BrainFlow Setup ===
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'
board_id = BoardIds.CYTON_DAISY_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)

channel_names = {
    1: "ECG", 2: "PPG", 3: "PCG", 4: "EMG1", 5: "EMG2",
    6: "MYOMETER", 7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP", 10: "OXYGEN",
    11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13", 14: "EEG CH14",
    15: "EEG CH15", 16: "EEG CH16"
}

async def main():
    global board, board_initialized
    board = BoardShim(board_id, params)

    try:
        print("🔄 Preparing BrainFlow session...")
        board.prepare_session()

        # Use same gain config as GUI
        gui_gain_config = (
            'x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000X'
            'xQ010000XxW010000XxE010000XxR010000XxT010000XxY010000XxU010000XxI010000X'
        )
        board.config_board(gui_gain_config)
        time.sleep(0.5)

        board.start_stream()
        time.sleep(1)
        board_initialized = True
        print("✅ Streaming started")

        # SSL Setup
        ip = '10.42.0.1'
        port = 5555
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cert_path = os.path.join(current_dir, "cert.pem")
        key_path = os.path.join(current_dir, "key.pem")

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)

        async with websockets.serve(eeg_handler, ip, port, ssl=ssl_context):
            print(f"🔒 Secure WebSocket running at wss://{ip}:{port}")
            await asyncio.Future()
    except BrainFlowError as e:
        print("🚨 BrainFlow error:", e)
    except Exception as e:
        print("🚨 Unexpected error:", e)
    finally:
        cleanup()

if __name__ == '__main__':
    asyncio.run(main())
