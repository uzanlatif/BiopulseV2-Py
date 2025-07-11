import asyncio
import websockets
import json
import time
import signal
import sys
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

board = None
board_initialized = False
is_running = True  # Flag to control graceful shutdown

# --- Graceful Shutdown ---
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

# --- HR Calculation Functions ---
def bandpass_filter(sig, fs, low, high):
    b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, sig)

def pan_tompkins_hr(ecg, fs):
    def pipeline(x):
        b, a = butter(1, [5 / (0.5 * fs), 15 / (0.5 * fs)], btype='band')
        x = filtfilt(b, a, x)
        x = np.convolve(x, np.array([1, 2, 0, -2, -1]) / 8, mode='same')
        x = x ** 2
        x = np.convolve(x, np.ones(int(0.15 * fs)) / int(0.15 * fs), mode='same')
        return x
    out = pipeline(ecg)
    peaks, _ = find_peaks(out, distance=int(0.2 * fs), height=np.mean(out))
    return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else '--'

def calculate_ppg_hr(ppg, fs):
    filtered = bandpass_filter(ppg, fs, 0.5, 5)
    peaks, _ = find_peaks(filtered, distance=int(0.5 * fs), prominence=0.8)
    return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else '--'

def calculate_pcg_hr(pcg, fs):
    filtered = bandpass_filter(pcg, fs, 20, 45)
    envelope = np.abs(hilbert(filtered))
    peaks, _ = find_peaks(envelope, distance=int(0.6 * fs), prominence=1.0)
    return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else '--'

# --- Channel and Board Info ---
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

# --- WebSocket Handler ---
async def eeg_handler(websocket, path):
    print("🔌 Client connected")
    try:
        sampling_rate = board.get_sampling_rate(board_id)
        interval = 1.0 / sampling_rate
        send_interval = 1.0 / 125  # send at 125Hz

        while is_running:
            raw_data = board.get_board_data(3)
            if raw_data.shape[1] == 0:
                await asyncio.sleep(0.5)
                continue

            timestamp_now = time.time()

            # --- Collect Signal Data ---
            sensor_signals = {}
            for ch in eeg_channels:
                label = channel_names.get(ch, f"CH{ch}")
                samples = raw_data[ch]
                sensor_signals[label] = [
                    {
                        "y": int(samples[i]),
                        "__timestamp__": timestamp_now - (len(samples) - i - 1) * interval
                    }
                    for i in range(len(samples))
                ]

            # --- Compute Heart Rates ---
            hr_values = {}
            try:
                ecg_ch = [k for k, v in channel_names.items() if v == "ECG"][0]
                hr_values["ECG"] = pan_tompkins_hr(raw_data[ecg_ch], sampling_rate)
            except:
                hr_values["ECG"] = '--'

            try:
                ppg_ch = [k for k, v in channel_names.items() if v == "PPG"][0]
                hr_values["PPG"] = calculate_ppg_hr(raw_data[ppg_ch], sampling_rate)
            except:
                hr_values["PPG"] = '--'

            try:
                pcg_ch = [k for k, v in channel_names.items() if v == "PCG"][0]
                hr_values["PCG"] = calculate_pcg_hr(raw_data[pcg_ch], sampling_rate)
            except:
                hr_values["PCG"] = '--'

            # --- Send to Client ---
            payload = {
                "signals": sensor_signals,
                "heartrate": hr_values
            }
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(send_interval)

    except websockets.ConnectionClosed:
        print("❌ Client disconnected")
    except Exception as e:
        print("🚨 Handler error:", e)

# --- Main Entry Point ---
async def main():
    global board, board_initialized
    board = BoardShim(board_id, params)

    try:
        print("🔄 Preparing BrainFlow session...")
        board.prepare_session()

        # Configure gain if needed
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
