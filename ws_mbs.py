import asyncio
import websockets
import json
import time
import signal
import sys
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

# === Global setup ===
board = None
board_initialized = False

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

# === EEG/WebSocket handler ===
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        sampling_rate = board.get_sampling_rate(board_id)
        interval = 1.0 / sampling_rate

        while True:
            raw_data = board.get_board_data(50)  # latest 50 samples
            if raw_data.shape[1] == 0:
                print("[⚠️ WARNING] No data received from board yet.")
                await asyncio.sleep(0.5)
                continue

            sensor_data = {}
            timestamp_now = time.time()

            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                samples = raw_data[ch]

                sensor_data[label] = [
                    {
                        "y": int(samples[i]),  # ADC raw value
                        "__timestamp__": timestamp_now - (len(samples) - i - 1) * interval
                    }
                    for i in range(len(samples))
                ]

            await websocket.send(json.dumps(sensor_data))
            await asyncio.sleep(0.3)
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# === BrainFlow Config ===
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # update if needed
board_id = BoardIds.CYTON_DAISY_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)

channel_names = {
    1: "ECG", 2: "PPG", 3: "PCG", 4: "EMG1", 5: "EMG2",
    6: "MYOMETER", 7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP", 10: "OXYGEN",
    11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13", 14: "EEG CH14",
    15: "EEG CH15", 16: "EEG CH16"
}

# === Main WebSocket server (non-SSL) ===
async def main():
    global board, board_initialized
    board = BoardShim(board_id, params)

    try:
        print("🔄 Preparing BrainFlow session...")
        board.prepare_session()

        # Apply gain config matching GUI setup (gain = 6 for ECG)
        gain_config = (
            'x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000X'
            'xQ010000XxW010000XxE010000XxR010000XxT010000XxY010000XxU010000XxI010000X'
        )
        board.config_board(gain_config)
        time.sleep(0.5)

        board.start_stream()
        time.sleep(1)
        board_initialized = True
        print("✅ Streaming started")

        ip = '10.42.0.1'  # update to match your RPi/server IP
        port = 5555

        async with websockets.serve(eeg_handler, ip, port):
            print(f"🌐 WebSocket Server running at ws://{ip}:{port}")
            await asyncio.Future()
    except BrainFlowError as e:
        print("🚨 BrainFlow error:", e)
    except Exception as e:
        print("🚨 Unexpected error:", e)
    finally:
        cleanup()

if __name__ == '__main__':
    asyncio.run(main())
