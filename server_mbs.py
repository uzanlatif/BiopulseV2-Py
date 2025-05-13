# eeg_server_raspi.py

import asyncio
import websockets
import json
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import iirnotch, filtfilt
import signal
import sys

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # Pastikan ini sesuai (cek pakai `ls /dev/tty*`)

board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)
eeg_channels = BoardShim.get_eeg_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)

notch_filter_enabled = True
buffer_size = 256

# Setup board
def start_board():
    board.prepare_session()
    board.config_board('x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000XxQ010000XxW010000X')
    board.start_stream()
    print("✅ OpenBCI Stream Started")

# Notch filter
def notch_filter(data, freq=60.0, fs=250.0, quality=30):
    b, a = iirnotch(freq / (0.5 * fs), quality)
    return filtfilt(b, a, data)

# WebSocket handler
async def eeg_handler(websocket, path):
    print("🌐 Client connected")
    try:
        while True:
            data = board.get_current_board_data(buffer_size)
            output = {}

            for ch in eeg_channels:
                ch_data = data[ch, :]
                if notch_filter_enabled:
                    ch_data = notch_filter(ch_data, 60, sampling_rate)
                output[f'channel_{ch}'] = ch_data.tolist()

            await websocket.send(json.dumps(output))
            await asyncio.sleep(0.1)  # 10Hz
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")

# Main server runner
async def main():
    start_board()
    async with websockets.serve(eeg_handler, "0.0.0.0", 8765):
        print("🚀 WebSocket Server running at ws://0.0.0.0:8765")
        await asyncio.Future()

# Clean shutdown
def shutdown(signal_received, frame):
    print("🛑 Stopping stream and releasing session...")
    board.stop_stream()
    board.release_session()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# Run
if __name__ == "__main__":
    asyncio.run(main())
