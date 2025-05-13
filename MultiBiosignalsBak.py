import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import iirnotch, filtfilt
import csv
import datetime
from PyQt5.QtGui import QFont
from scipy.fft import fft
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys


# Configure BrainFlowInputParams
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # Adjust this to your actual device port

# Initialize and prepare the board
board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
board.prepare_session()
board.config_board('x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000XxQ010000XxW010000X')
board.start_stream()

# Get EEG/ECG channels
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

channel_names = {
    1: "ECG",
    2: "PPG",
    3: "PCG",
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

# Plotting area setup
right_column_layout = QtWidgets.QVBoxLayout()
plot_area = pg.GraphicsLayoutWidget()
right_column_layout.addWidget(plot_area)
main_layout.addLayout(right_column_layout)

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
        eeg_data = data[channel_index, :].flatten()
        eeg_data_buffers[channel_name] = np.roll(eeg_data_buffers[channel_name], -len(eeg_data))
        eeg_data_buffers[channel_name][-len(eeg_data):] = eeg_data

        if notch_checkbox.isChecked():
            eeg_data_buffers[channel_name] = notch_filter(eeg_data_buffers[channel_name], 60, fs)

        if fft_checkbox.isChecked():
            freq_data = np.abs(np.fft.fft(eeg_data_buffers[channel_name]))[:buffer_size // 2]
            curves[channel_name].setData(freq_data)
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

            
# Ensure all other parts of your script are properly defining and using these channel indices

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
    selected_channels = [item.text() for item in channel_selector.selectedItems()]
    eeg_data_buffers = {channel: np.zeros(buffer_size) for channel in selected_channels}
    update_plot_layout()

# Connect buttons to functions
channel_selector.itemSelectionChanged.connect(update_selected_channels)
start_logging_button.clicked.connect(start_logging)
stop_logging_button.clicked.connect(stop_logging)
start_button.clicked.connect(lambda: timer.start(20))
stop_button.clicked.connect(timer.stop)
timer.timeout.connect(update_plot)
close_button.clicked.connect(close_app)
restart_button.clicked.connect(restart_connection)

win.show()
app.exec_()

