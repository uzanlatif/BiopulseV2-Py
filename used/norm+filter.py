import asyncio
import websockets
import json
import time
import signal
import sys
import numpy as np
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, hilbert
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

# --- Setup ---
board = None
board_initialized = False
is_running = True
board_id = BoardIds.CYTON_DAISY_BOARD.value

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'

eeg_channels = BoardShim.get_eeg_channels(board_id)

channel_names = {
    1: "ECG", 2: "PPG", 3: "PCG", 4: "EMG1", 5: "EMG2",
    6: "MYOMETER", 7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP", 10: "OXYGEN",
    11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13", 14: "EEG CH14",
    15: "EEG CH15", 16: "EEG CH16"
}

# --- Signal Handling ---
def signal_handler(sig, frame):
    global is_running
    print("\n🛑 Signal received, cleaning up...")
    is_running = False

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

# --- Filters and HR Estimators ---
def notch_filter(data, freq, fs, quality=30):
    nyq = 0.5 * fs
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, data)

def normalize(data):
    if np.ptp(data) == 0:
        return np.zeros_like(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def bandpass_filter(sig, fs, low, high):
    b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, sig)

def pan_tompkins_hr(ecg, fs):
    try:
        b, a = butter(1, [5 / (0.5 * fs), 15 / (0.5 * fs)], btype='band')
        x = filtfilt(b, a, ecg)
        x = np.convolve(x, np.array([1, 2, 0, -2, -1]) / 8, mode='same')
        x = x ** 2
        x = np.convolve(x, np.ones(int(0.15 * fs)) / int(0.15 * fs), mode='same')
        peaks, _ = find_peaks(x, distance=int(0.2 * fs), height=np.mean(x))
        return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else '--'
    except:
        return '--'

def estimate_hr_from_ppg(ppg, fs):
    try:
        f = bandpass_filter(ppg, fs, 0.5, 5)
        norm_f = (f - np.mean(f)) / np.std(f)
        peaks, _ = find_peaks(norm_f, distance=int(0.5 * fs), prominence=0.8)
        return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else '--'
    except:
        return '--'

def estimate_hr_from_pcg(pcg, fs):
    try:
        f = bandpass_filter(pcg, fs, 20, 45)
        env = np.abs(hilbert(f))
        norm_env = (env - np.mean(env)) / np.std(env)
        peaks, _ = find_peaks(norm_env, distance=int(0.6 * fs), prominence=1.0)
        return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else '--'
    except:
        return '--'

# --- WebSocket Handler ---
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        fs = BoardShim.get_sampling_rate(board_id)
        interval = 1.0 / fs
        send_interval = 1.0 / 125  # ~125Hz

        while is_running:
            raw_data = board.get_board_data()
            if raw_data.shape[1] == 0:
                await asyncio.sleep(0.1)
                continue

            sensor_data = {}
            timestamp_now = time.time()

            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                samples = raw_data[ch]

                # Filter + Normalize
                filtered = notch_filter(samples, 60.0, fs)
                normed = normalize(filtered)

                sensor_data[label] = [
                    {
                        "x": timestamp_now - (len(normed) - i - 1) * interval,
                        "y": round(float(normed[i]), 6)
                    }
                    for i in range(len(normed))
                ]

            # --- Heart Rate Estimation ---
            hr_values = {}
            hr_values['ECG'] = pan_tompkins_hr(raw_data[1], fs)
            hr_values['PPG'] = estimate_hr_from_ppg(raw_data[2], fs)
            hr_values['PCG'] = estimate_hr_from_pcg(raw_data[3], fs)

            # Final payload with timestamp
            payload = {
                "__timestamp__": timestamp_now,
                "HR": hr_values,
                **sensor_data
            }

            await websocket.send(json.dumps(payload))
            await asyncio.sleep(send_interval)

    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# --- Main Server Loop ---
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

        ip = '0.0.0.0'
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
        cleanup()

if __name__ == '__main__':
    asyncio.run(main())
