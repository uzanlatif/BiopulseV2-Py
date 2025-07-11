import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import iirnotch, filtfilt, butter, find_peaks, hilbert
import csv
import datetime
from PyQt5.QtGui import QFont
from scipy.fft import fft
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys

# --- BrainFlow Setup ---
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'
board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
board.prepare_session()
board.config_board('x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000XxQ010000XxW010000X')
board.start_stream()

# --- Channel Info ---
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
channel_names = {
    1: "ECG",
    2: "PCG",
    3: "PPG",
    4: "EMG1",
    5: "EMG2",
    6: "MYOMETER",
    7: "SPIRO",
    8: "TEMPERATURE",
    9: "NIBP",
    10: "OXYGEN",
    11: "EEG CH11",
    12: "EEG CH12",
    13: "EEG CH13",
    14: "EEG CH14",
    15: "EEG CH15",
    16: "EEG CH16"
}

# Set up the main application window
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle('MULTIBIOSIGNALS')
win.resize(1200, 800)
win.setStyleSheet("background-color: #F4F2F0; color: #000000;")
win.showFullScreen()  # Set default to full screen
main_layout = QtWidgets.QHBoxLayout(win)

# --- Left Column ---
# Channel selector setup
left_column_layout = QtWidgets.QVBoxLayout()
left_column_layout.setSpacing(12)

# Create a label for the image
image_label = QLabel()
pixmap = QPixmap('icons/meta.png')
image_label.setPixmap(pixmap)
image_label.setScaledContents(True)  # Make the image scale with the label size
image_label.setMaximumHeight(40)  # Optionally set a maximum height for the image
image_label.setMaximumWidth(180)  # Optionally set a maximum height for the image

title_label = QtWidgets.QLabel("BioPulse\n=========")  # Judul aplikasi untuk kolom kiri
title_label.setAlignment(QtCore.Qt.AlignCenter)  # Opsional: Tengahkan teks label
title_label_font = QFont("Arial", 25)
#title_label.setStyleSheet("color:red;")
title_label_font.setBold(True)
title_label.setFont(title_label_font)
title_label.setContentsMargins(0,0,0,0)

# Add channel selection header
channel_header_label = QtWidgets.QLabel("MULTIBIOSIGNALS \n \nSelect Channel")
channel_header_font = QFont("Arial", 14)
channel_header_font.setBold(True)
channel_header_label.setFont(channel_header_font)
channel_header_label.setContentsMargins(0, 0, 0, 0)

channel_selector = QtWidgets.QListWidget()
for i in eeg_channels:
    name = channel_names.get(i, f"Channel {i}")  # Gunakan nama custom atau default
    item = QtWidgets.QListWidgetItem(name)
    item.setData(QtCore.Qt.UserRole, i)
    channel_selector.addItem(item)

channel_selector.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
channel_selector.setMaximumWidth(150)
channel_selector.setMinimumHeight(400)

#list left column
left_column_layout.addWidget(image_label) 
left_column_layout.addWidget(title_label)  # Tambahkan label ke tata letak
left_column_layout.addWidget(channel_header_label)
left_column_layout.addWidget(channel_selector)
left_column_layout.addStretch(1) 
main_layout.addLayout(left_column_layout)

# --- Right Column ---
# Plotting area setup
right_column_layout = QtWidgets.QVBoxLayout()
plot_area = pg.GraphicsLayoutWidget()
right_column_layout.addWidget(plot_area)
main_layout.addLayout(right_column_layout)

# Heart rate UI setup
hr_label = QtWidgets.QLabel("HR (ECG): -- bpm | HR (PPG): -- bpm | HR (PCG): -- bpm")
hr_label_font = QFont("Arial", 16)
hr_label.setFont(hr_label_font)
hr_label.setAlignment(QtCore.Qt.AlignLeft)
right_column_layout.addWidget(hr_label)

# Control buttons and checkboxes
control_layout = QtWidgets.QHBoxLayout()
start_button = QtWidgets.QPushButton("Start")
stop_button = QtWidgets.QPushButton("Stop")
start_logging_button = QtWidgets.QPushButton("Start Logging")
stop_logging_button = QtWidgets.QPushButton("Stop Logging")
fft_checkbox = QtWidgets.QCheckBox("Show FFT")
notch_checkbox = QtWidgets.QCheckBox("60Hz Notch Filter")
close_button = QtWidgets.QPushButton("HOME")
restart_button = QtWidgets.QPushButton("Restart")
control_layout.addWidget(restart_button)
control_layout.addWidget(start_button)
control_layout.addWidget(stop_button)
control_layout.addWidget(start_logging_button)
control_layout.addWidget(stop_logging_button)
control_layout.addWidget(fft_checkbox)
control_layout.addWidget(notch_checkbox)
control_layout.addWidget(close_button)
right_column_layout.addLayout(control_layout)

# Plot curves dictionary and data buffers
curves = {}
buffer_size = 1200
eeg_data_buffers = {}
timer = QtCore.QTimer()
plots = {}  # Store separate plot widgets for each channel if needed
selected_channels = []
file_handle = None
logging_active = False
current_hr_values = {'ECG': '--', 'PPG': '--', 'PCG': '--'}

# --- Utility Functions ---
def update_hr_label():
    hr_label.setText(f"HR (ECG): {current_hr_values['ECG']} bpm | HR (PPG): {current_hr_values['PPG']} bpm | HR (PCG): {current_hr_values['PCG']} bpm")

def bandpass_filter(sig, fs, low, high):
    b, a = butter(2, [low / (fs/2), high / (fs/2)], btype='band')
    return filtfilt(b, a, sig)

def pan_tompkins_hr(ecg, fs):
    def pipeline(x):
        b, a = butter(1, [5/(0.5*fs), 15/(0.5*fs)], btype='band')
        x = filtfilt(b, a, x)
        x = np.convolve(x, np.array([1, 2, 0, -2, -1])/8, mode='same')
        x = x ** 2
        x = np.convolve(x, np.ones(int(0.15*fs))/int(0.15*fs), mode='same')
        return x
    out = pipeline(ecg)
    peaks, _ = find_peaks(out, distance=int(0.2*fs), height=np.mean(out))
    return int(60.0 / np.mean(np.diff(peaks) / fs)) if len(peaks) > 1 else '--'

# Fungsi untuk restart koneksi ke OpenBCI
def restart_connection():
    global board
    try:
        print("🔁 Restarting OpenBCI connection...")
        board.stop_stream()
        board.release_session()
    except Exception as e:
        print("Warning during cleanup:", e)
    
    # Buat ulang board dan mulai stream
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
    board.prepare_session()
    board.config_board('x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000XxQ010000XxW010000X')
    board.start_stream()
    print("✅ Connection restarted.")

# Notch filter function
def notch_filter(data, freq, fs, quality=30):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, quality)
    y = filtfilt(b, a, data)
    return y

# Function to close the app
def close_app():
    board.stop_stream()
    board.release_session()
    app.quit()

# Variabel global untuk handle file
file_handle = None
logging_active = False

def start_logging():
    global file_handle, logging_active

    # Membuat timestamp saat ini dan memformat nama file default
    current_time = datetime.datetime.now()
    default_filename = current_time.strftime("MultiBio_%m_%d_%H_%M_%S.csv")
    default_filepath = f"./DataLog/{default_filename}"  # Default filepath in the current directory

    # Membuka dialog file dengan nama file default sebagai pilihan awal
    options = QtWidgets.QFileDialog.Options()
    filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        None,  # Menggunakan None atau window utama sebagai parent, contoh 'win' jika ada
        "Start Logging Data",
        default_filepath,  # Menggunakan default filepath
        "CSV Files (*.csv);;All Files (*)",
        options=options
    )

    # Mengecek jika pengguna menyediakan nama file atau membatalkan dialog
    if filename:
        # Membuka file dengan nama yang dipilih atau default
        file_handle = open(filename, 'a', newline='')  # Buka file dalam mode append
        logging_active = True
        writer = csv.writer(file_handle)
        # Tulis header jika file baru
        if file_handle.tell() == 0:  # Cek apakah file kosong
            writer.writerow(['Time'] + selected_channels)
        print(f"Logging started in {filename}")
    else:
        # Handle jika pengguna membatalkan dialog
        logging_active = False
        print("Logging canceled. No file was created.")



def stop_logging():
    global file_handle, logging_active
    if file_handle:
        file_handle.close()
        file_handle = None
        logging_active = False
        print("Logging stopped.")

def update_plot():
    global eeg_data_buffers
    data = board.get_board_data()
    fs = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)

    for channel_name in selected_channels:
        channel_index = channel_selector.findItems(channel_name, QtCore.Qt.MatchExactly)[0].data(QtCore.Qt.UserRole)
        eeg_data = data[channel_index, -buffer_size:].flatten()
        eeg_data_buffers[channel_name] = np.roll(eeg_data_buffers[channel_name], -len(eeg_data))
        eeg_data_buffers[channel_name][-len(eeg_data):] = eeg_data

        if notch_checkbox.isChecked():
            eeg_data_buffers[channel_name] = notch_filter(eeg_data_buffers[channel_name], 60, fs)

        if fft_checkbox.isChecked():
            freq_data = np.abs(np.fft.fft(eeg_data_buffers[channel_name]))[:buffer_size // 2]
            curves[channel_name].setData(freq_data)
        signal = eeg_data_buffers[channel_name]
        if channel_name == "PPG":
            signal = -signal
            signal = signal - np.min(signal)
            signal = signal / np.max(signal)
            signal = signal * 100
            curves[channel_name].setData(signal)
        elif channel_name in ["ECG", "PCG", "EMG1", "EMG2", "EEG11", "EEG12", "EEG13", "EEG14", "EEG15", "EEG16"]:
            signal = signal - np.min(signal)
            signal = signal / np.max(signal)
            signal = signal * 100
            curves[channel_name].setData(signal)
        elif channel_name == "MYOMETER":    # Newton
            signal = (signal - 109840)/30000
            curves[channel_name].setData(signal) 
        elif channel_name == "SPIRO":
            signal = signal - 1100000
            signal = 0.010698 * signal - 9.3359e-9 * signal**2
            curves[channel_name].setData(signal) # miliLiter/second
        elif channel_name == "TEMPERATURE":  # Celcius
            signal = -signal
            signal = signal - np.min(signal)
            curves[channel_name].setData(signal)
        elif channel_name == "NIBP":    # mmHg
            signal = signal - np.min(signal)
            curves[channel_name].setData(signal)
        elif channel_name == "OXYGEN":    # %O2
            signal = -signal
            signal = signal - np.min(signal)
            curves[channel_name].setData(signal)
        else:
            curves[channel_name].setData(eeg_data_buffers[channel_name])

    if logging_active and file_handle:
        writer = csv.writer(file_handle)
        # Tulis data terbaru ke file
        for i in range(data.shape[1]):  # Asumsikan data memiliki bentuk [channel, samples]
            row = [i]  # Indeks waktu/sample
            for channel_name in selected_channels:  # Memperbaiki penggunaan variable loop yang salah
                if i < len(eeg_data_buffers[channel_name]):  # Pastikan tidak melebihi panjang buffer
                    row.append(eeg_data_buffers[channel_name][i])
                else:
                    row.append('')  # Menambahkan placeholder jika data tidak tersedia
            writer.writerow(row)

def update_hr():
    fs = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
    
    for name, buffer in eeg_data_buffers.items():
        if len(buffer) < fs * 2:
            current_hr_values[name] = '--'
            continue

        if name == "ECG":
            try:
                current_hr_values['ECG'] = pan_tompkins_hr(buffer, fs)
            except:
                current_hr_values['ECG'] = '--'

        elif name == "PPG":
            try:
                f = bandpass_filter(buffer, fs, 0.5, 5)
                norm_f = (f - np.mean(f)) / np.std(f)
                peaks, _ = find_peaks(norm_f, distance=int(0.5 * fs), prominence=0.8)
                if len(peaks) > 1:
                    hr = 60.0 / np.mean(np.diff(peaks) / fs)
                    current_hr_values['PPG'] = int(hr)
                else:
                    current_hr_values['PPG'] = '--'
            except:
                current_hr_values['PPG'] = '--'

        elif name == "PCG":
            try:
                # Bandpass dulu untuk S1-S2 range
                f = bandpass_filter(buffer, fs, 20, 45)

                # Ambil envelope menggunakan Hilbert
                analytic = hilbert(f)
                envelope = np.abs(analytic)

                # Normalisasi envelope
                norm_env = (envelope - np.mean(envelope)) / np.std(envelope)

                # Cari puncak di envelope (deteksi S1)
                peaks, _ = find_peaks(norm_env, distance=int(0.6 * fs), prominence=1.0)

                if len(peaks) > 1:
                    hr = 60.0 / np.mean(np.diff(peaks) / fs)
                    current_hr_values['PCG'] = int(hr)
                else:
                    current_hr_values['PCG'] = '--'
            except:
                current_hr_values['PCG'] = '--'

    update_hr_label()


hr_timer = QtCore.QTimer()
hr_timer.timeout.connect(update_hr)            

def update_plot_layout():
    global selected_channels, curves, plot_area
    num_channels = len(selected_channels)
    rows = num_channels // 2 if num_channels > 6 else num_channels
    cols = 2 if num_channels > 6 else 1

    plot_area.clear()
    for idx, channel_name in enumerate(selected_channels):
        row = idx // cols
        col = idx % cols
        plot_widget = plot_area.addPlot(row=row, col=col, title=channel_name)
        plot_widget.setXRange(0, buffer_size)
        curve = plot_widget.plot()
        curves[channel_name] = curve

# Update selected channels and layout when selection changes
def update_selected_channels():
    global selected_channels, curves, eeg_data_buffers
#asli    selected_channels = [item.text() for item in channel_selector.selectedItems()]
    selected_channels = {
    item.text(): item.data(QtCore.Qt.UserRole)
    for item in channel_selector.selectedItems()
}
    eeg_data_buffers = {channel: np.zeros(buffer_size) for channel in selected_channels}
    update_plot_layout()

def start_all():
    update_selected_channels()
    update_plot()
    timer.start(20)
    hr_timer.start(1000)


# Connect buttons to functions
channel_selector.itemSelectionChanged.connect(update_selected_channels)
start_logging_button.clicked.connect(start_logging)
stop_logging_button.clicked.connect(stop_logging)
start_button.clicked.connect(start_all)
stop_button.clicked.connect(timer.stop)
timer.timeout.connect(update_plot)
close_button.clicked.connect(close_app)
restart_button.clicked.connect(restart_connection)

win.show()
app.exec_()
