import asyncio
import websockets
import json
import time
import signal
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

# --- Global Variables ---
board = None
board_initialized = False
is_running = True

# --- Signal Handlers ---
def signal_handler(sig, frame):
    global is_running
    print("\n🛑 Signal received, shutting down...")
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

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- HR Calculation Utilities ---
def bandpass_filter(sig, fs, low, high):
    b, a = butter(2, [low / (fs/2), high / (fs/2)], btype='band')
    return filtfilt(b, a, sig)

def pan_tompkins_hr(ecg, fs):
    try:
        b, a = butter(1, [5/(0.5*fs), 15/(0.5*fs)], btype='band')
        x = filtfilt(b, a, ecg)
        x = np.convolve(x, np.array([1, 2, 0, -2, -1])/8, mode='same')
        x = x ** 2
        x = np.convolve(x, np.ones(int(0.15*fs))/int(0.15*fs), mode='same')
        peaks, _ = find_peaks(x, distance=int(0.2*fs), height=np.mean(x))
        return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else None
    except:
        return None

def ppg_hr(ppg, fs):
    try:
        f = bandpass_filter(ppg, fs, 0.5, 5)
        norm_f = (f - np.mean(f)) / np.std(f)
        peaks, _ = find_peaks(norm_f, distance=int(0.5 * fs), prominence=0.8)
        if len(peaks) > 1:
            return int(60.0 / np.mean(np.diff(peaks) / fs))
    except:
        return None
    return None

def pcg_hr(pcg, fs):
    try:
        f = bandpass_filter(pcg, fs, 20, 45)
        envelope = np.abs(hilbert(f))
        norm_env = (envelope - np.mean(envelope)) / np.std(envelope)
        peaks, _ = find_peaks(norm_env, distance=int(0.6 * fs), prominence=1.0)
        if len(peaks) > 1:
            return int(60.0 / np.mean(np.diff(peaks) / fs))
    except:
        return None
    return None

# --- EEG Handler ---
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        sampling_rate = board.get_sampling_rate(board_id)
        interval = 1.0 / sampling_rate
        send_interval = 1.0 / 125  # send 125Hz
        buffer_len = int(sampling_rate * 3)

        while is_running:
            raw_data = board.get_current_board_data(buffer_len)
            if raw_data.shape[1] == 0:
                await asyncio.sleep(0.5)
                continue

            timestamp_now = time.time()
            sensor_data = {}
            hr_data = {}

            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                samples = raw_data[ch].astype(np.float32)
                signal = samples.copy()

                # --- Heart rate per channel ---
                if label == "ECG":
                    hr_data["ECG"] = pan_tompkins_hr(signal, sampling_rate)
                elif label == "PPG":
                    hr_data["PPG"] = ppg_hr(signal, sampling_rate)
                elif label == "PCG":
                    hr_data["PCG"] = pcg_hr(signal, sampling_rate)

                # --- Normalization per channel ---
                if label == "PPG":
                    signal = -signal
                    signal -= np.min(signal)
                    signal /= np.max(signal) if np.max(signal) != 0 else 1
                    signal *= 100
                elif label in ["ECG", "PCG", "EMG1", "EMG2", "EEG CH11", "EEG CH12", "EEG CH13", "EEG CH14", "EEG CH15", "EEG CH16"]:
                    signal -= np.min(signal)
                    signal /= np.max(signal) if np.max(signal) != 0 else 1
                    signal *= 100
                elif label == "MYOMETER":
                    signal = (signal - 109840) / 30000
                elif label == "SPIRO":
                    signal = signal - 1100000
                    signal = 0.010698 * signal - 9.3359e-9 * signal**2
                elif label == "TEMPERATURE":
                    signal = -signal
                    signal -= np.min(signal)
                elif label == "NIBP":
                    signal -= np.min(signal)
                elif label == "OXYGEN":
                    signal = -signal
                    signal -= np.min(signal)

                # --- Compose signal data with timestamps ---
                sensor_data[label] = [
                    {
                        "y": float(signal[i]),
                        "__timestamp__": timestamp_now - (len(signal) - i - 1) * interval
                    }
                    for i in range(len(signal))
                ]

            # --- Send via WebSocket ---
            response = {
                "signals": sensor_data,
                "heartrate": hr_data
            }

            await websocket.send(json.dumps(response))
            await asyncio.sleep(send_interval)

    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# --- BrainFlow Setup ---
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # ganti sesuai serial port kamu
board_id = BoardIds.CYTON_DAISY_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)

channel_names = {
    1: "ECG", 2: "PPG", 3: "PCG", 4: "EMG1", 5: "EMG2",
    6: "MYOMETER", 7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP", 10: "OXYGEN",
    11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13", 14: "EEG CH14",
    15: "EEG CH15", 16: "EEG CH16"
}

# --- Main App ---
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
        cleanup()

if __name__ == '__main__':
    asyncio.run(main())
