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

# Signal handler for graceful shutdown
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
            await asyncio.sleep(0.3)
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# Config and EEG channels
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'
board_id = BoardIds.CYTON_DAISY_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)
channel_names = {
    1: "LEAD_I", 2: "LEAD_II", 3: "LEAD_III", 4: "AVR", 5: "AVL", 6: "AVF",
    7: "V1", 8: "V2", 9: "V3", 10: "V4", 11: "V5", 12: "V6"
}

async def main():
    global board, board_initialized
    board = BoardShim(board_id, params)
    try:
        print("🔄 Preparing BrainFlow session...")
        board.prepare_session()
        board.start_stream()
        board_initialized = True
        print("✅ Streaming started")

        ip = '172.30.81.62'
        port = 8888
        async with websockets.serve(eeg_handler, ip, port):
            print(f"🌐 WebSocket Server running at ws://{ip}:{port}")
            await asyncio.Future()  # run forever
    except BrainFlowError as e:
        print("🚨 BrainFlow setup failed:", e)
    except Exception as e:
        print("🚨 Unexpected error:", e)
    finally:
        cleanup()

if __name__ == '__main__':
    asyncio.run(main())
