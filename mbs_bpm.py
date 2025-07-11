import asyncio
import websockets
import json
import time
import signal
import sys
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from scipy.signal import butter, filtfilt, find_peaks, hilbert
import numpy as np

# --- Global Setup ---
board = None
board_initialized = False
is_running = True
board_id = BoardIds.CYTON_DAISY_BOARD.value

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'

eeg_channels = BoardShim.get_eeg_channels(board_id)

channel_names = {
    1: "ECG", 2: "PCG", 3: "PPG", 4: "EMG1", 5: "EMG2",
    6: "MYOMETER", 7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP", 10: "OXYGEN",
    11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13", 14: "EEG CH14",
    15: "EEG CH15", 16: "EEG CH16"
}

def signal_handler(sig, frame):
    global is_running
    print("\n🛑 Signal received, cleaning up...")
    is_running = False

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# --- Filtering ---
def bandpass_filter(sig, fs, low, high):
    b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, sig)

# --- HR Extractors ---
def hr_from_ecg(ecg, fs):
    try:
        b, a = butter(1, [5 / (0.5 * fs), 15 / (0.5 * fs)], btype='band')
        ecg = filtfilt(b, a, ecg)
        ecg = np.convolve(ecg, np.array([1, 2, 0, -2, -1]) / 8, mode='same')
        ecg = ecg ** 2
        ecg = np.convolve(ecg, np.ones(int(0.15 * fs)) / int(0.15 * fs), mode='same')
        peaks, _ = find_peaks(ecg, distance=int(0.2 * fs), height=np.mean(ecg))
        return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else None
    except:
        return None

def hr_from_ppg(ppg, fs):
    try:
        f = bandpass_filter(ppg, fs, 0.5, 5)
        norm_f = (f - np.mean(f)) / np.std(f)
        peaks, _ = find_peaks(norm_f, distance=int(0.5 * fs), prominence=0.8)
        return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else None
    except:
        return None

def hr_from_pcg(pcg, fs):
    try:
        f = bandpass_filter(pcg, fs, 20, 45)
        envelope = np.abs(hilbert(f))
        norm_env = (envelope - np.mean(envelope)) / np.std(envelope)
        peaks, _ = find_peaks(norm_env, distance=int(0.6 * fs), prominence=1.0)
        return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else None
    except:
        return None

# --- WebSocket Handler ---
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        fs = BoardShim.get_sampling_rate(board_id)
        interval = 1.0 / fs
        send_interval = 1.0 / 125  # adjust if needed

        while is_running:
            raw_data = board.get_board_data(3)
            if raw_data.shape[1] == 0:
                await asyncio.sleep(0.5)
                continue

            sensor_data = {}
            timestamp_now = time.time()

            hr_data = {'ECG': None, 'PPG': None, 'PCG': None}

            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                samples = raw_data[ch]
                sensor_data[label] = [
                    {
                        "y": float(samples[i]),
                        "__timestamp__": timestamp_now - (len(samples) - i - 1) * interval
                    }
                    for i in range(len(samples))
                ]

                # Extract HR if applicable
                if label == "ECG":
                    hr_data['ECG'] = hr_from_ecg(samples, fs)
                elif label == "PPG":
                    hr_data['PPG'] = hr_from_ppg(samples, fs)
                elif label == "PCG":
                    hr_data['PCG'] = hr_from_pcg(samples, fs)

            # Send combined data
            await websocket.send(json.dumps({
                "signals": sensor_data,
                "heartrate": hr_data,
                "timestamp": timestamp_now
            }))

            await asyncio.sleep(send_interval)

    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# --- Main ---
async def main():
    global board, board_initialized
    board = BoardShim(board_id, params)

    try:
        print("🔄 Preparing BrainFlow session...")
        board.prepare_session()

        gain_config = (
            'x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000X'
            'xQ010000XxW010000XxE010000XxR010000XxT010000XxY010000XxU010000XxI010000X'
        )
        board.config_board(gain_config)
        time.sleep(0.5)

        board.start_stream()
        board_initialized = True
        print("✅ Streaming started")

        ip = '10.42.0.1'
        port = 5555
        async with websockets.serve(eeg_handler, ip, port):
            print(f"🌐 WebSocket Server running at ws://{ip}:{port}")
            while is_running:
                await asyncio.sleep(0.1)

    except BrainFlowError as e:
        print("🚨 BrainFlow error:", e)
    except Exception as e:
        print("🚨 Unexpected error:", e)
    finally:
        if board_initialized:
            board.stop_stream()
            board.release_session()
        print("🧹 Cleaned up session")

if __name__ == '__main__':
    asyncio.run(main())
