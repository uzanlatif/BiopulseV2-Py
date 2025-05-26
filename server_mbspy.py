import asyncio
import websockets
import json
import time
import signal
import sys
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

# Global board instance
board = None
board_initialized = False

# Handle Ctrl+C or SIGTERM for graceful shutdown
def signal_handler(sig, frame):
    print("\n🛑 Signal received, cleaning up...")
    cleanup()
    sys.exit(0)

# Cleanup BrainFlow session
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

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# WebSocket handler
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        sampling_rate = board.get_sampling_rate(board_id)
        interval = 1.0 / sampling_rate

        while True:
            raw_data = board.get_current_board_data(50)
            sensor_data = {}
            timestamp_now = time.time()

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
            await asyncio.sleep(0.1)  # Send data every 100ms
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# BrainFlow parameters
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # Change this if your USB dongle is at a different port
board_id = BoardIds.CYTON_DAISY_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)

# Optional: Rename channels for better readability
channel_names = {
    0: "EEG CH1", 1: "EEG CH2", 2: "EEG CH3", 3: "EEG CH4",
    4: "EEG CH5", 5: "EEG CH6", 6: "EEG CH7", 7: "EEG CH8",
    8: "EEG CH9", 9: "EEG CH10", 10: "EEG CH11", 11: "EEG CH12",
    12: "EEG CH13", 13: "EEG CH14", 14: "EEG CH15", 15: "EEG CH16"
}

# Main async function
async def main():
    global board, board_initialized
    board = BoardShim(board_id, params)

    try:
        print("🔄 Preparing BrainFlow session...")
        board.prepare_session()
        board.start_stream()
        board_initialized = True
        print("✅ EEG stream started")

        ip = '0.0.0.0'  # Or replace with actual IP like '172.30.81.62'
        port = 9999

        async with websockets.serve(eeg_handler, ip, port):
            print(f"🌐 WebSocket Server running at ws://{ip}:{port}")
            await asyncio.Future()  # Keeps the server running
    except BrainFlowError as e:
        print("🚨 BrainFlow error:", e)
    except Exception as e:
        print("🚨 Unexpected error:", e)
    finally:
        cleanup()

if __name__ == '__main__':
    asyncio.run(main())
