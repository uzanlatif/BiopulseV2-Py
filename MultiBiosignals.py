import os
import sys
import numpy as np
import csv
import datetime
from scipy.signal import iirnotch, filtfilt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget, QFileDialog, QCheckBox, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.fft import fft

# Initialize BrainFlow and parameters
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'

try:
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
    board.prepare_session()
    board.config_board('x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000XxQ010000XxW010000X')
    board.start_stream()
except Exception as e:
    print("❌ Board Error:", e)
    QtWidgets.QMessageBox.critical(None, "Board Error", str(e))
    sys.exit(1)

eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

channel_names = {
    1: "ECG", 2: "PPG", 3: "PCG", 4: "EMG1", 5: "EMG2",
    6: "MYOMETER", 7: "SPIRO", 8: "TEMPERATURE", 9: "NIBP",
    10: "OXYGEN", 11: "EEG CH11", 12: "EEG CH12", 13: "EEG CH13",
    14: "EEG CH14", 15: "EEG CH15", 16: "EEG CH16"
}

# GUI setup
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle('MULTIBIOSIGNALS')
win.resize(1200, 800)
win.setStyleSheet("background-color: #F4F2F0; color: #000000;")
win.showFullScreen()
main_layout = QtWidgets.QHBoxLayout(win)

left_column_layout = QtWidgets.QVBoxLayout()
left_column_layout.setSpacing(12)

image_label = QLabel()
pixmap = QPixmap('icons/meta.png')
image_label.setPixmap(pixmap)
image_label.setScaledContents(True)
image_label.setMaximumHeight(40)
image_label.setMaximumWidth(180)

title_label = QtWidgets.QLabel("BioPulse\n=========")
title_label.setAlignment(Qt.AlignCenter)
title_label_font = QFont("Arial", 25)
title_label_font.setBold(True)
title_label.setFont(title_label_font)

channel_header_label = QtWidgets.QLabel("MULTIBIOSIGNALS \n \nSelect Channel")
channel_header_font = QFont("Arial", 14)
channel_header_font.setBold(True)
channel_header_label.setFont(channel_header_font)

channel_selector = QtWidgets.QListWidget()
for i in eeg_channels:
    name = channel_names.get(i, f"Channel {i}")
    item = QtWidgets.QListWidgetItem(name)
    item.setData(QtCore.Qt.UserRole, i)
    channel_selector.addItem(item)
channel_selector.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
channel_selector.setMaximumWidth(150)
channel_selector.setMinimumHeight(400)

left_column_layout.addWidget(image_label)
left_column_layout.addWidget(title_label)
left_column_layout.addWidget(channel_header_label)
left_column_layout.addWidget(channel_selector)
left_column_layout.addStretch(1)
main_layout.addLayout(left_column_layout)

right_column_layout = QtWidgets.QVBoxLayout()
plot_area = pg.GraphicsLayoutWidget()
right_column_layout.addWidget(plot_area)
main_layout.addLayout(right_column_layout)

control_layout = QtWidgets.QHBoxLayout()
start_button = QPushButton("Start")
stop_button = QPushButton("Stop")
start_logging_button = QPushButton("Start Logging")
stop_logging_button = QPushButton("Stop Logging")
fft_checkbox = QCheckBox("Show FFT")
notch_checkbox = QCheckBox("60Hz Notch Filter")
close_button = QPushButton("HOME")
restart_button = QPushButton("Restart")
for btn in [restart_button, start_button, stop_button, start_logging_button, stop_logging_button, fft_checkbox, notch_checkbox, close_button]:
    control_layout.addWidget(btn)
right_column_layout.addLayout(control_layout)

# Data structures
curves = {}
buffer_size = 1200
eeg_data_buffers = {}
timer = QTimer()
plots = {}
selected_channels = []
file_handle = None
logging_active = False

# Functions
def restart_connection():
    global board
    try:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
    except Exception as e:
        print("Warning during cleanup:", e)
    try:
        board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        board.prepare_session()
        board.config_board('x1060100Xx2010000Xx3010000Xx4060000Xx5060000Xx6010000Xx7010000Xx8010000XxQ010000XxW010000X')
        board.start_stream()
        print("✅ Connection restarted.")
    except Exception as e:
        print("❌ Reconnection failed:", e)

def notch_filter(data, freq, fs, quality=30):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, quality)
    return filtfilt(b, a, data)

def close_app():
    try:
        board.stop_stream()
        board.release_session()
    except Exception as e:
        print("Error on shutdown:", e)
    app.quit()

def start_logging():
    global file_handle, logging_active
    current_time = datetime.datetime.now()
    default_filename = current_time.strftime("MultiBio_%m_%d_%H_%M_%S.csv")
    default_filepath = f"./DataLog/{default_filename}"
    os.makedirs(os.path.dirname(default_filepath), exist_ok=True)

    filename, _ = QFileDialog.getSaveFileName(None, "Start Logging Data", default_filepath, "CSV Files (*.csv);;All Files (*)")
    if filename:
        file_handle = open(filename, 'a', newline='')
        logging_active = True
        writer = csv.writer(file_handle)
        if file_handle.tell() == 0:
            writer.writerow(['Time'] + selected_channels)
        print(f"✅ Logging started: {filename}")
    else:
        logging_active = False
        print("⚠️ Logging canceled.")

def stop_logging():
    global file_handle, logging_active
    if file_handle:
        file_handle.close()
        file_handle = None
        logging_active = False
        print("🛑 Logging stopped.")

def update_plot():
    global eeg_data_buffers
    try:
        data = board.get_board_data()
        fs = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)

        for channel_name in selected_channels:
            items = channel_selector.findItems(channel_name, QtCore.Qt.MatchExactly)
            if not items:
                continue
            channel_index = items[0].data(QtCore.Qt.UserRole)
            eeg_data = data[channel_index, :].flatten()
            eeg_data_buffers[channel_name] = np.roll(eeg_data_buffers[channel_name], -len(eeg_data))
            eeg_data_buffers[channel_name][-len(eeg_data):] = eeg_data

            display_data = eeg_data_buffers[channel_name]
            if notch_checkbox.isChecked():
                try:
                    display_data = notch_filter(display_data, 60, fs)
                except Exception as e:
                    print("⚠️ Notch filter error:", e)

            if fft_checkbox.isChecked():
                try:
                    freq_data = np.abs(fft(display_data))[:buffer_size // 2]
                    curves[channel_name].setData(freq_data)
                except Exception as e:
                    print("⚠️ FFT error:", e)
            else:
                curves[channel_name].setData(display_data)

        if logging_active and file_handle:
            writer = csv.writer(file_handle)
            for i in range(data.shape[1]):
                row = [i]
                for channel_name in selected_channels:
                    if i < len(eeg_data_buffers[channel_name]):
                        row.append(eeg_data_buffers[channel_name][i])
                    else:
                        row.append('')
                writer.writerow(row)
    except Exception as e:
        print("⚠️ Plot update error:", e)

def update_plot_layout():
    global selected_channels, curves, plot_area
    num_channels = len(selected_channels)
    rows = num_channels // 2 if num_channels > 6 else num_channels
    cols = 2 if num_channels > 6 else 1

    plot_area.clear()
    curves.clear()
    for idx, channel_name in enumerate(selected_channels):
        row = idx // cols
        col = idx % cols
        plot_widget = plot_area.addPlot(row=row, col=col, title=channel_name)
        plot_widget.setXRange(0, buffer_size)
        curve = plot_widget.plot()
        curves[channel_name] = curve

def update_selected_channels():
    global selected_channels, eeg_data_buffers
    selected_channels = [item.text() for item in channel_selector.selectedItems()]
    eeg_data_buffers = {channel: np.zeros(buffer_size) for channel in selected_channels}
    update_plot_layout()

# Connect buttons
channel_selector.itemSelectionChanged.connect(update_selected_channels)
start_logging_button.clicked.connect(start_logging)
stop_logging_button.clicked.connect(stop_logging)
start_button.clicked.connect(lambda: timer.start(20))
stop_button.clicked.connect(timer.stop)
timer.timeout.connect(update_plot)
close_button.clicked.connect(close_app)
restart_button.clicked.connect(restart_connection)

# Show window
win.show()
app.exec_()
