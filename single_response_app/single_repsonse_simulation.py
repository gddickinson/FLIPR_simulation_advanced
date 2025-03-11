import sys
import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QGroupBox, QFormLayout, QLabel, QComboBox, QDoubleSpinBox,
                           QSpinBox, QCheckBox, QPushButton, QTabWidget, QTextEdit,
                           QFileDialog, QMessageBox, QSplitter, QSizePolicy, QRadioButton)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QTextCursor
from datetime import datetime

# Configure logging
LOG_FOLDER = 'logs'
os.makedirs(LOG_FOLDER, exist_ok=True)
log_filename = os.path.join(LOG_FOLDER, f'agonist_simulator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AgonistResponseSimulator')

class MatplotlibCanvas(FigureCanvas):
    """Canvas for Matplotlib figures in the GUI"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class DebugConsole(QTextEdit):
    """Custom console for displaying log messages"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)

        # Custom styling for the console
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #DDDDDD;
                font-family: Consolas, Monaco, monospace;
                font-size: 10pt;
            }
        """)

    def append_message(self, msg, level='INFO'):
        """Add a message to the console with appropriate formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if level == 'ERROR':
            self.append(f'<span style="color:red;">[{timestamp}] ERROR: {msg}</span>')
        elif level == 'WARNING':
            self.append(f'<span style="color:#FFA500;">[{timestamp}] WARNING: {msg}</span>')
        elif level == 'DEBUG':
            self.append(f'<span style="color:#AAAAAA;">[{timestamp}] DEBUG: {msg}</span>')
        else:
            self.append(f'<span style="color:#DDDDDD;">[{timestamp}] INFO: {msg}</span>')

        # Auto-scroll to bottom
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)

class SimulationThread(QThread):
    """Thread for running simulations without freezing the GUI"""
    progress_update = pyqtSignal(int)
    simulation_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            logger.info("Starting simulation on separate thread")

            # Generate time points
            time_interval = self.config.get('time_interval', 0.4)
            num_timepoints = self.config.get('num_timepoints', 450)
            total_time = time_interval * num_timepoints
            time_points = np.arange(0, total_time, time_interval)

            # Generate calcium response
            response = self.generate_calcium_response(
                time_points=time_points,
                baseline=self.config.get('baseline', 500),
                peak=self.config.get('peak', 1000),
                rise_rate=self.config.get('rise_rate', 0.05),
                decay_rate=self.config.get('decay_rate', 0.05),
                agonist_time=self.config.get('agonist_addition_time', 10)
            )

            # Apply errors if enabled
            if self.config.get('apply_errors', False):
                response = self.apply_errors(response, time_points)

            # Apply noise and photobleaching
            noisy_response = self.add_realistic_noise(
                response,
                self.config.get('read_noise', 20),
                self.config.get('background', 100),
                self.config.get('photobleaching_rate', 0.0005)
            )

            if self.config.get('simulate_df_f0', False):
                # Calculate baseline period
                baseline_end = int(self.config.get('agonist_addition_time', 10) / self.config.get('time_interval', 0.4))
                if baseline_end > 0 and baseline_end < len(noisy_response):
                    # Calculate F0 as mean of baseline period
                    f0 = np.mean(noisy_response[:baseline_end])
                    # Calculate DF/F0
                    if self.config.get('df_f0_as_percent', True):
                        df_f0_response = ((noisy_response - f0) / f0) * 100  # Express as percentage
                    else:
                        df_f0_response = (noisy_response - f0) / f0  # Express as ratio

                    # Store both raw and DF/F0 signals
                    results = {
                        'time_points': time_points,
                        'raw_signal': response,
                        'signal': noisy_response,
                        'df_f0_signal': df_f0_response,
                        'config': self.config,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    # Handle case with insufficient baseline points
                    self.error_occurred.emit("Insufficient baseline points for DF/F0 calculation")
                    return
            else:
                # Original results without DF/F0
                results = {
                    'time_points': time_points,
                    'raw_signal': response,
                    'signal': noisy_response,
                    'config': self.config,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            logger.info("Simulation completed successfully")
            self.simulation_complete.emit(results)

        except Exception as e:
            logger.error(f"Error in simulation thread: {str(e)}", exc_info=True)
            self.error_occurred.emit(str(e))

    def generate_calcium_response(self, time_points, baseline, peak, rise_rate, decay_rate, agonist_time):
        """Generate a calcium response curve based on the parameters"""
        response = np.zeros_like(time_points, dtype=float)

        # Get agonist type and concentration
        agonist_type = self.config.get('agonist_type', 'ATP')
        concentration = self.config.get('concentration', 100)

        # Calculate dose-response factor
        dose_factor = self.calculate_dose_response_factor(agonist_type, concentration)

        # Adjust peak height based on dose-response
        adjusted_peak = baseline + (peak - baseline) * dose_factor

        # Set baseline values
        agonist_idx = int(agonist_time / (time_points[1] - time_points[0]))
        response[:agonist_idx] = baseline

        # Calculate peak time (seconds after agonist addition)
        peak_time_offset = 5  # 5 seconds after agonist addition
        peak_idx = agonist_idx + int(peak_time_offset / (time_points[1] - time_points[0]))

        # Generate response curve with rise and decay
        if agonist_idx < len(time_points):
            # Rising phase
            if peak_idx > agonist_idx:
                rise_indices = np.arange(agonist_idx, min(peak_idx + 1, len(time_points)))
                rise_duration = peak_idx - agonist_idx
                if rise_duration > 0:
                    for i in rise_indices:
                        progress = (i - agonist_idx) / rise_duration
                        response[i] = baseline + (adjusted_peak - baseline) * (np.sin(progress * np.pi/2) ** 2)

            # Decay phase
            if peak_idx < len(time_points) - 1:
                decay_indices = np.arange(peak_idx, len(time_points))
                decay_times = time_points[decay_indices] - time_points[peak_idx]
                response[decay_indices] = baseline + (adjusted_peak - baseline) * np.exp(-decay_rate * decay_times)

        return response

    def apply_errors(self, signal, time_points):
        """Apply selected error types to the signal"""
        modified_signal = signal.copy()

        # Get error settings
        error_types = self.config.get('error_types', {})

        # Apply each enabled error
        if error_types.get('random_spikes', {}).get('enabled', False):
            # Add random spikes
            params = error_types.get('random_spikes', {})
            num_spikes = params.get('count', 3)
            spike_amplitude = params.get('amplitude', 1000)
            spike_width = params.get('width', 3)

            for _ in range(num_spikes):
                spike_pos = np.random.randint(0, len(signal))
                for i in range(max(0, spike_pos - spike_width), min(len(signal), spike_pos + spike_width + 1)):
                    distance = abs(i - spike_pos) / spike_width
                    modified_signal[i] += spike_amplitude * np.exp(-4 * distance ** 2)

        if error_types.get('signal_dropouts', {}).get('enabled', False):
            # Add signal dropouts
            params = error_types.get('signal_dropouts', {})
            num_dropouts = params.get('count', 1)
            dropout_length = params.get('length', 10)
            dropout_factor = 1.0 - params.get('factor', 0.2)

            for _ in range(num_dropouts):
                dropout_pos = np.random.randint(0, len(signal) - dropout_length)
                modified_signal[dropout_pos:dropout_pos + dropout_length] *= dropout_factor

        if error_types.get('baseline_drift', {}).get('enabled', False):
            # Add baseline drift
            params = error_types.get('baseline_drift', {})
            drift_direction = 1 if params.get('direction', 'Rising') == 'Rising' else -1
            drift_magnitude = params.get('magnitude', 500)

            drift = drift_direction * drift_magnitude * np.linspace(0, 1, len(signal))
            modified_signal += drift

        if error_types.get('oscillating_baseline', {}).get('enabled', False):
            # Add oscillating baseline
            params = error_types.get('oscillating_baseline', {})
            frequency = params.get('frequency', 0.05)
            amplitude = params.get('amplitude', 300)

            oscillation = amplitude * np.sin(2 * np.pi * frequency * time_points)
            modified_signal += oscillation

        if error_types.get('delayed_response', {}).get('enabled', False):
            # Delay the response
            params = error_types.get('delayed_response', {})
            delay_seconds = params.get('delay_seconds', 5)

            delay_points = int(delay_seconds / (time_points[1] - time_points[0]))

            if delay_points > 0:
                agonist_idx = int(self.config.get('agonist_addition_time', 10) / (time_points[1] - time_points[0]))
                baseline = self.config.get('baseline', 500)

                # Create new array with delay
                delayed_signal = np.ones_like(signal) * baseline

                # Copy the response but shifted
                src_indices = np.arange(agonist_idx, len(signal))
                dst_indices = np.minimum(src_indices + delay_points, len(signal) - 1)

                # Apply delay
                delayed_signal[dst_indices] = signal[src_indices[:len(dst_indices)]]
                modified_signal = delayed_signal

        if error_types.get('incomplete_decay', {}).get('enabled', False):
            # Response doesn't return to baseline
            params = error_types.get('incomplete_decay', {})
            elevation_factor = params.get('elevation_factor', 0.5)

            agonist_idx = int(self.config.get('agonist_addition_time', 10) / (time_points[1] - time_points[0]))
            if agonist_idx < len(signal) - 10:
                # Find peak after agonist addition
                post_addition = signal[agonist_idx:]
                peak_idx = agonist_idx + np.argmax(post_addition)

                if peak_idx < len(signal) - 10:
                    # Calculate baseline
                    baseline = self.config.get('baseline', 500)

                    # Get peak height
                    peak_height = signal[peak_idx] - baseline

                    # Calculate incomplete decay for points after peak
                    decay_indices = np.arange(peak_idx + 1, len(signal))
                    for i in decay_indices:
                        # Calculate how far into decay we are (0 to 1)
                        progress = (i - peak_idx) / (len(signal) - peak_idx)
                        # Add elevation that decreases as we get further from peak
                        original_value = modified_signal[i]
                        distance_from_baseline = original_value - baseline
                        # More elevation for points closer to baseline
                        factor = 1 - (distance_from_baseline / peak_height)
                        elevation = peak_height * elevation_factor * max(0, factor)
                        modified_signal[i] = original_value + elevation

        if error_types.get('extra_noise', {}).get('enabled', False):
            # Add extra Gaussian noise
            params = error_types.get('extra_noise', {})
            noise_std = params.get('std', 100)

            # Generate noise
            noise = np.random.normal(0, noise_std, len(signal))

            # Add to signal
            modified_signal += noise

        if error_types.get('sudden_jump', {}).get('enabled', False):
            # Add a sudden jump in the signal
            params = error_types.get('sudden_jump', {})
            position_pct = params.get('position_pct', 0.7)
            jump_magnitude = params.get('magnitude', 500)

            # Calculate jump position
            jump_idx = int(position_pct * len(signal))

            # Apply jump to all points after the position
            if jump_idx > 0 and jump_idx < len(signal):
                modified_signal[jump_idx:] += jump_magnitude

        if error_types.get('signal_cutout', {}).get('enabled', False):
            # Replace a section of the signal with zeros or baseline
            params = error_types.get('signal_cutout', {})
            start_pct = params.get('start_pct', 0.3)
            duration_pct = params.get('duration_pct', 0.2)
            replace_with_baseline = params.get('replace_with_baseline', True)

            # Calculate start and end indices
            start_idx = int(start_pct * len(signal))
            duration = int(duration_pct * len(signal))
            end_idx = min(start_idx + duration, len(signal))

            if replace_with_baseline:
                # Use baseline value
                baseline = self.config.get('baseline', 500)
                modified_signal[start_idx:end_idx] = baseline
            else:
                # Use zeros
                modified_signal[start_idx:end_idx] = 0

        if error_types.get('exp_drift', {}).get('enabled', False):
            # Add exponential drift
            params = error_types.get('exp_drift', {})
            direction = params.get('direction', 'Upward')
            magnitude = params.get('magnitude', 1000)
            rate = params.get('rate', 3)

            # Calculate exponential drift
            t = np.linspace(0, 1, len(signal))
            if direction == 'Upward':
                drift = magnitude * (np.exp(rate * t) - 1) / (np.exp(rate) - 1)
            else:  # Downward
                drift = magnitude * (np.exp(rate * (1 - t)) - 1) / (np.exp(rate) - 1)

            # Add drift to signal
            modified_signal += drift

        return modified_signal

    def add_realistic_noise(self, signal, read_noise=20, background=100, photobleaching_rate=0.0005):
        """Add realistic noise to the fluorescence signal"""
        # Ensure signal is float
        signal = signal.astype(float)

        # Create copy for noise addition
        noisy_signal = signal.copy()

        # Add Gaussian read noise
        noisy_signal += np.random.normal(0, read_noise, signal.shape)

        # Add background
        noisy_signal += background

        # Apply photobleaching
        time = np.arange(len(signal))
        photobleaching = np.exp(-photobleaching_rate * time)
        noisy_signal *= photobleaching

        return np.maximum(noisy_signal, 0).round(2)  # Ensure non-negative values and round

    def calculate_dose_response_factor(self, agonist_type, concentration):
        """Calculate the dose-response factor based on agonist type and concentration"""
        # Define EC50 values for different agonists (in µM)
        ec50_values = {
            "ATP": 100.0,
            "UTP": 150.0,
            "Ionomycin": 0.5,
            "Buffer": float('inf'),  # Will result in near-zero response
            "Custom": 100.0  # Default for custom agonist
        }

        # Define Hill coefficients for different agonists
        hill_coefficients = {
            "ATP": 1.5,
            "UTP": 1.3,
            "Ionomycin": 2.0,
            "Buffer": 1.0,
            "Custom": 1.5  # Default for custom agonist
        }

        # Get EC50 and Hill coefficient for the selected agonist
        ec50 = ec50_values.get(agonist_type, 100.0)
        hill = hill_coefficients.get(agonist_type, 1.5)

        # Special case for Buffer (should give minimal response)
        if agonist_type == "Buffer" or concentration <= 0:
            return 0.05  # Minimal response

        # Calculate response using Hill equation
        response_factor = (concentration ** hill) / (ec50 ** hill + concentration ** hill)

        # Ensure factor is between 0 and 1
        return max(0, min(1, response_factor))

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()

        # Add a flag to track whether parameters are being auto-updated
        self.auto_updating = False

        # Add a flag to track whether parameters have been manually overridden
        self.params_overridden = False

        # Create debug console first
        self.debug_console = DebugConsole()

        # Initialize UI
        self.init_ui()

        # Store last results
        self.last_results = None

        # Log application start
        logger.info("Agonist Response Simulator application started")
        self.debug_console.append_message("Agonist Response Simulator started")

    def init_ui(self):
        """Initialize the UI components"""
        self.setWindowTitle("Agonist Response Simulator")
        self.setGeometry(100, 100, 1000, 700)

        # Create tab widget
        self.tabs = QTabWidget()

        # Add tabs
        self.tabs.addTab(self.create_simulation_tab(), "Simulation")
        self.tabs.addTab(self.create_debug_tab(), "Debug Console")

        # Set as central widget
        self.setCentralWidget(self.tabs)

        # Add status bar
        self.statusBar().showMessage("Ready")

    def create_simulation_tab(self):
        """Create the main simulation tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Create horizontal splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - simulation controls
        control_panel = QWidget()
        control_layout = QVBoxLayout()

        # Agonist selection
        agonist_group = QGroupBox("Agonist Settings")
        agonist_layout = QFormLayout()

        self.agonist_combo = QComboBox()
        self.agonist_combo.addItems(["ATP", "UTP", "Ionomycin", "Buffer", "Custom"])
        self.agonist_combo.currentTextChanged.connect(self.on_agonist_changed)
        agonist_layout.addRow("Agonist Type:", self.agonist_combo)

        self.concentration_spin = QDoubleSpinBox()
        self.concentration_spin.setRange(0.001, 1000)
        self.concentration_spin.setValue(100)
        self.concentration_spin.setDecimals(3)
        self.concentration_spin.setSingleStep(10)
        agonist_layout.addRow("Concentration (μM):", self.concentration_spin)

        agonist_group.setLayout(agonist_layout)
        control_layout.addWidget(agonist_group)

        self.concentration_spin.valueChanged.connect(self.on_concentration_changed)

        # Signal parameters
        signal_group = QGroupBox("Signal Parameters")
        signal_layout = QFormLayout()

        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setRange(0, 2000)
        self.baseline_spin.setValue(500)
        self.baseline_spin.setSingleStep(50)
        signal_layout.addRow("Baseline:", self.baseline_spin)

        self.peak_spin = QDoubleSpinBox()
        self.peak_spin.setRange(0, 5000)
        self.peak_spin.setValue(1000)
        self.peak_spin.setSingleStep(100)
        signal_layout.addRow("Peak Height:", self.peak_spin)

        self.rise_rate_spin = QDoubleSpinBox()
        self.rise_rate_spin.setRange(0.01, 0.5)
        self.rise_rate_spin.setValue(0.05)
        self.rise_rate_spin.setDecimals(3)
        self.rise_rate_spin.setSingleStep(0.01)
        signal_layout.addRow("Rise Rate:", self.rise_rate_spin)

        self.decay_rate_spin = QDoubleSpinBox()
        self.decay_rate_spin.setRange(0.001, 0.2)
        self.decay_rate_spin.setValue(0.05)
        self.decay_rate_spin.setDecimals(3)
        self.decay_rate_spin.setSingleStep(0.01)
        signal_layout.addRow("Decay Rate:", self.decay_rate_spin)

        self.df_f0_check = QCheckBox("Simulate DF/F0 Instead of Raw Fluorescence")
        self.df_f0_check.setToolTip("When enabled, signals will be calculated as (F-F0)/F0 relative to baseline")
        signal_layout.addRow("", self.df_f0_check)

        # Connect checkbox to enable/disable format selection
        self.df_f0_check.stateChanged.connect(lambda state: self.df_f0_format_group.setEnabled(state))

        # Add radio buttons for DF/F0 format selection
        self.df_f0_format_group = QGroupBox("DF/F0 Format")
        self.df_f0_format_group.setEnabled(False)  # Initially disabled until DF/F0 is checked
        df_f0_format_layout = QHBoxLayout()

        self.df_f0_percent_radio = QRadioButton("Percentage (%)")
        self.df_f0_percent_radio.setChecked(True)  # Default option
        self.df_f0_percent_radio.setToolTip("Display DF/F0 as percentage (e.g., 10%)")

        self.df_f0_ratio_radio = QRadioButton("Ratio")
        self.df_f0_ratio_radio.setToolTip("Display DF/F0 as decimal ratio (e.g., 0.1)")

        df_f0_format_layout.addWidget(self.df_f0_percent_radio)
        df_f0_format_layout.addWidget(self.df_f0_ratio_radio)
        self.df_f0_format_group.setLayout(df_f0_format_layout)
        signal_layout.addRow("", self.df_f0_format_group)

        # In the create_simulation_tab method, add this to the signal parameters group
        reset_params_btn = QPushButton("Reset Parameters")
        reset_params_btn.setToolTip("Reset parameters to automatic values based on agonist and concentration")
        reset_params_btn.clicked.connect(self.reset_parameter_override)
        signal_layout.addRow("", reset_params_btn)

        signal_group.setLayout(signal_layout)
        control_layout.addWidget(signal_group)

        # Acquisition parameters
        acq_group = QGroupBox("Acquisition Parameters")
        acq_layout = QFormLayout()

        self.num_timepoints_spin = QSpinBox()
        self.num_timepoints_spin.setRange(10, 1000)
        self.num_timepoints_spin.setValue(450)
        self.num_timepoints_spin.setSingleStep(50)
        acq_layout.addRow("Number of Timepoints:", self.num_timepoints_spin)

        self.time_interval_spin = QDoubleSpinBox()
        self.time_interval_spin.setRange(0.1, 5)
        self.time_interval_spin.setValue(0.4)
        self.time_interval_spin.setDecimals(2)
        self.time_interval_spin.setSingleStep(0.1)
        acq_layout.addRow("Time Interval (seconds):", self.time_interval_spin)

        self.agonist_time_spin = QDoubleSpinBox()
        self.agonist_time_spin.setRange(1, 50)
        self.agonist_time_spin.setValue(10)
        self.agonist_time_spin.setSingleStep(1)
        acq_layout.addRow("Agonist Addition Time (seconds):", self.agonist_time_spin)

        acq_group.setLayout(acq_layout)
        control_layout.addWidget(acq_group)

        # Noise parameters
        noise_group = QGroupBox("Noise Parameters")
        noise_layout = QFormLayout()

        self.read_noise_spin = QDoubleSpinBox()
        self.read_noise_spin.setRange(0, 500)
        self.read_noise_spin.setValue(20)
        self.read_noise_spin.setSingleStep(10)
        noise_layout.addRow("Read Noise:", self.read_noise_spin)

        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(0, 2000)
        self.background_spin.setValue(100)
        self.background_spin.setSingleStep(50)
        noise_layout.addRow("Background:", self.background_spin)

        self.photobleaching_spin = QDoubleSpinBox()
        self.photobleaching_spin.setRange(0, 0.05)
        self.photobleaching_spin.setValue(0.0005)
        self.photobleaching_spin.setDecimals(6)
        self.photobleaching_spin.setSingleStep(0.0005)
        noise_layout.addRow("Photobleaching Rate:", self.photobleaching_spin)

        # Reset button
        self.reset_noise_btn = QPushButton("Reset to Defaults")
        self.reset_noise_btn.clicked.connect(self.reset_noise_parameters)
        noise_layout.addRow("", self.reset_noise_btn)

        noise_group.setLayout(noise_layout)
        control_layout.addWidget(noise_group)

        # Error simulation
        error_group = QGroupBox("Error Simulation")
        error_layout = QVBoxLayout()

        self.apply_errors_check = QCheckBox("Apply Errors")
        error_layout.addWidget(self.apply_errors_check)

        # Use a tab widget for error parameters
        error_tabs = QTabWidget()

        # Random Spikes tab
        spikes_tab = QWidget()
        spikes_layout = QFormLayout()

        self.random_spikes_check = QCheckBox("Enable")
        spikes_layout.addRow("Random Spikes:", self.random_spikes_check)

        self.spikes_count_spin = QSpinBox()
        self.spikes_count_spin.setRange(1, 10)
        self.spikes_count_spin.setValue(3)
        spikes_layout.addRow("Number of Spikes:", self.spikes_count_spin)

        self.spikes_amplitude_spin = QDoubleSpinBox()
        self.spikes_amplitude_spin.setRange(100, 3000)
        self.spikes_amplitude_spin.setValue(1000)
        self.spikes_amplitude_spin.setSingleStep(100)
        spikes_layout.addRow("Spike Amplitude:", self.spikes_amplitude_spin)

        self.spikes_width_spin = QSpinBox()
        self.spikes_width_spin.setRange(1, 10)
        self.spikes_width_spin.setValue(3)
        spikes_layout.addRow("Spike Width:", self.spikes_width_spin)

        spikes_tab.setLayout(spikes_layout)
        error_tabs.addTab(spikes_tab, "Random Spikes")

        # Signal Dropouts tab
        dropouts_tab = QWidget()
        dropouts_layout = QFormLayout()

        self.signal_dropouts_check = QCheckBox("Enable")
        dropouts_layout.addRow("Signal Dropouts:", self.signal_dropouts_check)

        self.dropouts_count_spin = QSpinBox()
        self.dropouts_count_spin.setRange(1, 5)
        self.dropouts_count_spin.setValue(1)
        dropouts_layout.addRow("Number of Dropouts:", self.dropouts_count_spin)

        self.dropouts_length_spin = QSpinBox()
        self.dropouts_length_spin.setRange(5, 50)
        self.dropouts_length_spin.setValue(10)
        dropouts_layout.addRow("Dropout Length:", self.dropouts_length_spin)

        self.dropouts_factor_spin = QDoubleSpinBox()
        self.dropouts_factor_spin.setRange(0, 1)
        self.dropouts_factor_spin.setValue(0.2)
        self.dropouts_factor_spin.setDecimals(2)
        self.dropouts_factor_spin.setSingleStep(0.1)
        dropouts_layout.addRow("Remaining Signal:", self.dropouts_factor_spin)

        dropouts_tab.setLayout(dropouts_layout)
        error_tabs.addTab(dropouts_tab, "Signal Dropouts")

        # Baseline Drift tab
        drift_tab = QWidget()
        drift_layout = QFormLayout()

        self.baseline_drift_check = QCheckBox("Enable")
        drift_layout.addRow("Baseline Drift:", self.baseline_drift_check)

        self.drift_direction_combo = QComboBox()
        self.drift_direction_combo.addItems(["Rising", "Falling"])
        drift_layout.addRow("Direction:", self.drift_direction_combo)

        self.drift_magnitude_spin = QDoubleSpinBox()
        self.drift_magnitude_spin.setRange(100, 2000)
        self.drift_magnitude_spin.setValue(500)
        self.drift_magnitude_spin.setSingleStep(100)
        drift_layout.addRow("Magnitude:", self.drift_magnitude_spin)

        drift_tab.setLayout(drift_layout)
        error_tabs.addTab(drift_tab, "Baseline Drift")

        # Oscillating Baseline tab
        oscillation_tab = QWidget()
        oscillation_layout = QFormLayout()

        self.oscillating_baseline_check = QCheckBox("Enable")
        oscillation_layout.addRow("Oscillating Baseline:", self.oscillating_baseline_check)

        self.oscillation_frequency_spin = QDoubleSpinBox()
        self.oscillation_frequency_spin.setRange(0.01, 0.5)
        self.oscillation_frequency_spin.setValue(0.05)
        self.oscillation_frequency_spin.setDecimals(3)
        self.oscillation_frequency_spin.setSingleStep(0.01)
        oscillation_layout.addRow("Frequency:", self.oscillation_frequency_spin)

        self.oscillation_amplitude_spin = QDoubleSpinBox()
        self.oscillation_amplitude_spin.setRange(50, 1000)
        self.oscillation_amplitude_spin.setValue(300)
        self.oscillation_amplitude_spin.setSingleStep(50)
        oscillation_layout.addRow("Amplitude:", self.oscillation_amplitude_spin)

        oscillation_tab.setLayout(oscillation_layout)
        error_tabs.addTab(oscillation_tab, "Oscillating Baseline")

        # Delayed Response tab
        delay_tab = QWidget()
        delay_layout = QFormLayout()

        self.delayed_response_check = QCheckBox("Enable")
        delay_layout.addRow("Delayed Response:", self.delayed_response_check)

        self.delay_time_spin = QDoubleSpinBox()
        self.delay_time_spin.setRange(1, 200)
        self.delay_time_spin.setValue(5)
        self.delay_time_spin.setSingleStep(1)
        delay_layout.addRow("Delay Time (seconds):", self.delay_time_spin)

        delay_tab.setLayout(delay_layout)
        error_tabs.addTab(delay_tab, "Delayed Response")


        # Incomplete Decay tab
        incomplete_decay_tab = QWidget()
        incomplete_decay_layout = QFormLayout()

        self.incomplete_decay_check = QCheckBox("Enable")
        incomplete_decay_layout.addRow("Incomplete Decay:", self.incomplete_decay_check)

        self.elevation_factor_spin = QDoubleSpinBox()
        self.elevation_factor_spin.setRange(0.1, 0.9)
        self.elevation_factor_spin.setValue(0.5)
        self.elevation_factor_spin.setSingleStep(0.1)
        self.elevation_factor_spin.setDecimals(2)
        incomplete_decay_layout.addRow("Elevation Factor:", self.elevation_factor_spin)

        incomplete_decay_tab.setLayout(incomplete_decay_layout)
        error_tabs.addTab(incomplete_decay_tab, "Incomplete Decay")

        # Extra Noise tab
        extra_noise_tab = QWidget()
        extra_noise_layout = QFormLayout()

        self.extra_noise_check = QCheckBox("Enable")
        extra_noise_layout.addRow("Extra Noise:", self.extra_noise_check)

        self.noise_std_spin = QDoubleSpinBox()
        self.noise_std_spin.setRange(20, 500)
        self.noise_std_spin.setValue(100)
        self.noise_std_spin.setSingleStep(10)
        extra_noise_layout.addRow("Noise Std. Dev.:", self.noise_std_spin)

        extra_noise_tab.setLayout(extra_noise_layout)
        error_tabs.addTab(extra_noise_tab, "Extra Noise")

        # Sudden Jump tab
        sudden_jump_tab = QWidget()
        sudden_jump_layout = QFormLayout()

        self.sudden_jump_check = QCheckBox("Enable")
        sudden_jump_layout.addRow("Sudden Jump:", self.sudden_jump_check)

        self.jump_position_spin = QDoubleSpinBox()
        self.jump_position_spin.setRange(0.1, 0.9)
        self.jump_position_spin.setValue(0.7)
        self.jump_position_spin.setDecimals(2)
        self.jump_position_spin.setSingleStep(0.05)
        sudden_jump_layout.addRow("Position (% of trace):", self.jump_position_spin)

        self.jump_magnitude_spin = QDoubleSpinBox()
        self.jump_magnitude_spin.setRange(-2000, 2000)
        self.jump_magnitude_spin.setValue(500)
        self.jump_magnitude_spin.setSingleStep(100)
        sudden_jump_layout.addRow("Jump Magnitude:", self.jump_magnitude_spin)

        sudden_jump_tab.setLayout(sudden_jump_layout)
        error_tabs.addTab(sudden_jump_tab, "Sudden Jump")

        # Signal Cutout tab
        cutout_tab = QWidget()
        cutout_layout = QFormLayout()

        self.signal_cutout_check = QCheckBox("Enable")
        cutout_layout.addRow("Signal Cutout:", self.signal_cutout_check)

        self.cutout_start_spin = QDoubleSpinBox()
        self.cutout_start_spin.setRange(0.1, 0.9)
        self.cutout_start_spin.setValue(0.3)
        self.cutout_start_spin.setDecimals(2)
        self.cutout_start_spin.setSingleStep(0.05)
        cutout_layout.addRow("Start Position (% of trace):", self.cutout_start_spin)

        self.cutout_duration_spin = QDoubleSpinBox()
        self.cutout_duration_spin.setRange(0.05, 0.5)
        self.cutout_duration_spin.setValue(0.2)
        self.cutout_duration_spin.setDecimals(2)
        self.cutout_duration_spin.setSingleStep(0.05)
        cutout_layout.addRow("Duration (% of trace):", self.cutout_duration_spin)

        self.replace_baseline_check = QCheckBox("Replace with baseline (otherwise zeros)")
        self.replace_baseline_check.setChecked(True)
        cutout_layout.addRow("", self.replace_baseline_check)

        cutout_tab.setLayout(cutout_layout)
        error_tabs.addTab(cutout_tab, "Signal Cutout")

        # Exponential Drift tab
        exp_drift_tab = QWidget()
        exp_drift_layout = QFormLayout()

        self.exp_drift_check = QCheckBox("Enable")
        exp_drift_layout.addRow("Exponential Drift:", self.exp_drift_check)

        self.exp_direction_combo = QComboBox()
        self.exp_direction_combo.addItems(["Upward", "Downward"])
        exp_drift_layout.addRow("Direction:", self.exp_direction_combo)

        self.exp_magnitude_spin = QDoubleSpinBox()
        self.exp_magnitude_spin.setRange(100, 3000)
        self.exp_magnitude_spin.setValue(1000)
        self.exp_magnitude_spin.setSingleStep(100)
        exp_drift_layout.addRow("Maximum Magnitude:", self.exp_magnitude_spin)

        self.exp_rate_spin = QDoubleSpinBox()
        self.exp_rate_spin.setRange(1, 10)
        self.exp_rate_spin.setValue(3)
        self.exp_rate_spin.setDecimals(1)
        self.exp_rate_spin.setSingleStep(0.5)
        exp_drift_layout.addRow("Exponential Rate:", self.exp_rate_spin)

        exp_drift_tab.setLayout(exp_drift_layout)
        error_tabs.addTab(exp_drift_tab, "Exponential Drift")

        error_layout.addWidget(error_tabs)
        error_group.setLayout(error_layout)
        control_layout.addWidget(error_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self.run_simulation)

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_configuration)

        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self.load_configuration)

        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.save_config_btn)
        button_layout.addWidget(self.load_config_btn)

        control_layout.addLayout(button_layout)
        control_layout.addStretch()

        control_panel.setLayout(control_layout)

        # Right panel - visualization
        viz_panel = QWidget()
        viz_layout = QVBoxLayout()

        # Create matplotlib canvas
        self.canvas = MatplotlibCanvas(viz_panel, width=6, height=5, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, viz_panel)

        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas, 1)

        # Export controls
        export_layout = QHBoxLayout()

        self.export_plot_btn = QPushButton("Export Plot")
        self.export_plot_btn.clicked.connect(self.export_plot)

        self.export_data_btn = QPushButton("Export Data")
        self.export_data_btn.clicked.connect(self.export_data)
        self.export_data_btn.setEnabled(False)

        export_layout.addWidget(self.export_plot_btn)
        export_layout.addWidget(self.export_data_btn)
        export_layout.addStretch()

        viz_layout.addLayout(export_layout)

        # Signal information display
        info_group = QGroupBox("Signal Information")
        info_layout = QVBoxLayout()

        self.signal_info_label = QLabel("No simulation data available.")
        self.signal_info_label.setWordWrap(True)
        info_layout.addWidget(self.signal_info_label)

        info_group.setLayout(info_layout)
        viz_layout.addWidget(info_group)

        viz_panel.setLayout(viz_layout)

        # Add both panels to the splitter
        splitter.addWidget(control_panel)
        splitter.addWidget(viz_panel)

        # Set initial sizes
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)
        tab.setLayout(layout)


        # In the create_simulation_tab method:
        # Connect signal parameter changes to the override handler
        self.peak_spin.valueChanged.connect(self.on_parameter_changed)
        self.rise_rate_spin.valueChanged.connect(self.on_parameter_changed)
        self.decay_rate_spin.valueChanged.connect(self.on_parameter_changed)

        # In the create_simulation_tab method:
        self.concentration_spin.valueChanged.connect(self.on_concentration_changed)

        # Initialize by selecting ATP
        self.on_agonist_changed("ATP")

        return tab

    def create_debug_tab(self):
        """Create the debug console tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add console controls
        controls_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear Console")
        clear_btn.clicked.connect(self.debug_console.clear)

        log_level_combo = QComboBox()
        log_level_combo.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        log_level_combo.currentTextChanged.connect(self.set_log_level)

        save_log_btn = QPushButton("Save Log")
        save_log_btn.clicked.connect(self.save_log)

        controls_layout.addWidget(QLabel("Log Level:"))
        controls_layout.addWidget(log_level_combo)
        controls_layout.addWidget(clear_btn)
        controls_layout.addWidget(save_log_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)
        layout.addWidget(self.debug_console)

        tab.setLayout(layout)

        # Add initial message
        self.debug_console.append_message("Agonist Response Simulator started")

        return tab

    def set_log_level(self, level):
        """Set the logging level"""
        numeric_level = getattr(logging, level)
        logging.getLogger().setLevel(numeric_level)
        self.debug_console.append_message(f"Log level set to {level}")

    def save_log(self):
        """Save the debug console log to a file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Log",
                os.path.join(os.getcwd(), f"agonist_simulator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
                "Text Files (*.txt)"
            )

            if filename:
                with open(filename, 'w') as f:
                    f.write(self.debug_console.toPlainText())

                self.statusBar().showMessage(f"Log saved to {filename}")
                self.debug_console.append_message(f"Log saved to {filename}")
        except Exception as e:
            self.debug_console.append_message(f"Error saving log: {str(e)}", level='ERROR')

    def on_agonist_changed(self, agonist_type):
        """Update parameters based on selected agonist"""
        if self.params_overridden:
            # If parameters have been manually set, don't auto-update
            if hasattr(self, 'debug_console'):
                self.debug_console.append_message(f"Selected agonist: {agonist_type} (using manual parameters)")
            return

        # Set auto_updating flag to prevent recursive updates
        self.auto_updating = True

        # Default parameters for each agonist type
        if agonist_type == "ATP":
            self.baseline_spin.setValue(500)
            # Peak will be set based on concentration
            self.rise_rate_spin.setValue(0.05)
            self.decay_rate_spin.setValue(0.05)
            self.concentration_spin.setValue(100)
        elif agonist_type == "UTP":
            self.baseline_spin.setValue(500)
            # Peak will be set based on concentration
            self.rise_rate_spin.setValue(0.06)
            self.decay_rate_spin.setValue(0.04)
            self.concentration_spin.setValue(150)
        elif agonist_type == "Ionomycin":
            self.baseline_spin.setValue(500)
            # Peak will be set based on concentration
            self.rise_rate_spin.setValue(0.08)
            self.decay_rate_spin.setValue(0.01)
            self.concentration_spin.setValue(1)
        elif agonist_type == "Buffer":
            self.baseline_spin.setValue(500)
            # Peak will be set based on concentration
            self.rise_rate_spin.setValue(0.03)
            self.decay_rate_spin.setValue(0.03)
            self.concentration_spin.setValue(0)
        # Custom parameters are left as-is

        # Update peak based on concentration
        self.update_peak_from_concentration()

        # Reset auto_updating flag
        self.auto_updating = False

        # Only log if debug_console exists
        if hasattr(self, 'debug_console'):
            self.debug_console.append_message(f"Selected agonist: {agonist_type}")

    def reset_noise_parameters(self):
        """Reset noise parameters to their default values"""
        try:
            # Set default values
            self.read_noise_spin.setValue(20)
            self.background_spin.setValue(100)
            self.photobleaching_spin.setValue(0.0005)

            # Show confirmation message
            self.statusBar().showMessage("Noise parameters reset to defaults", 3000)
            self.debug_console.append_message("Noise parameters reset to defaults")

        except Exception as e:
            self.debug_console.append_message(f"Error resetting noise parameters: {str(e)}", level='ERROR')


    def get_simulation_config(self):
        """Get the current simulation configuration"""
        config = {
            'agonist_type': self.agonist_combo.currentText(),
            'concentration': self.concentration_spin.value(),
            'baseline': self.baseline_spin.value(),
            'peak': self.peak_spin.value(),
            'rise_rate': self.rise_rate_spin.value(),
            'decay_rate': self.decay_rate_spin.value(),
            'num_timepoints': self.num_timepoints_spin.value(),
            'time_interval': self.time_interval_spin.value(),
            'agonist_addition_time': self.agonist_time_spin.value(),
            'read_noise': self.read_noise_spin.value(),
            'background': self.background_spin.value(),
            'photobleaching_rate': self.photobleaching_spin.value(),
            'simulate_df_f0': self.df_f0_check.isChecked(),
            'df_f0_as_percent': self.df_f0_percent_radio.isChecked(),
            'apply_errors': self.apply_errors_check.isChecked(),
            'error_types': {
                'random_spikes': {
                    'enabled': self.random_spikes_check.isChecked(),
                    'count': self.spikes_count_spin.value(),
                    'amplitude': self.spikes_amplitude_spin.value(),
                    'width': self.spikes_width_spin.value()
                },
                'signal_dropouts': {
                    'enabled': self.signal_dropouts_check.isChecked(),
                    'count': self.dropouts_count_spin.value(),
                    'length': self.dropouts_length_spin.value(),
                    'factor': self.dropouts_factor_spin.value()
                },
                'baseline_drift': {
                    'enabled': self.baseline_drift_check.isChecked(),
                    'direction': self.drift_direction_combo.currentText(),
                    'magnitude': self.drift_magnitude_spin.value()
                },
                'oscillating_baseline': {
                    'enabled': self.oscillating_baseline_check.isChecked(),
                    'frequency': self.oscillation_frequency_spin.value(),
                    'amplitude': self.oscillation_amplitude_spin.value()
                },
                'delayed_response': {
                    'enabled': self.delayed_response_check.isChecked(),
                    'delay_seconds': self.delay_time_spin.value()
                },
                'incomplete_decay': {
                    'enabled': self.incomplete_decay_check.isChecked(),
                    'elevation_factor': self.elevation_factor_spin.value()
                },
                'extra_noise': {
                    'enabled': self.extra_noise_check.isChecked(),
                    'std': self.noise_std_spin.value()
                },
                'sudden_jump': {
                    'enabled': self.sudden_jump_check.isChecked(),
                    'position_pct': self.jump_position_spin.value(),
                    'magnitude': self.jump_magnitude_spin.value()
                },
                'signal_cutout': {
                    'enabled': self.signal_cutout_check.isChecked(),
                    'start_pct': self.cutout_start_spin.value(),
                    'duration_pct': self.cutout_duration_spin.value(),
                    'replace_with_baseline': self.replace_baseline_check.isChecked()
                },
                'exp_drift': {
                    'enabled': self.exp_drift_check.isChecked(),
                    'direction': self.exp_direction_combo.currentText(),
                    'magnitude': self.exp_magnitude_spin.value(),
                    'rate': self.exp_rate_spin.value()
                }
            }
        }

        return config

    def run_simulation(self):
        """Run a simulation with the current configuration"""
        try:
            # Get configuration from UI
            config = self.get_simulation_config()

            # Log simulation start
            self.debug_console.append_message(f"Starting simulation with {config['num_timepoints']} timepoints")

            # Create and start simulation thread
            self.sim_thread = SimulationThread(config)
            self.sim_thread.simulation_complete.connect(self.simulation_completed)
            self.sim_thread.error_occurred.connect(self.simulation_error)

            # Update UI
            self.run_btn.setEnabled(False)
            self.statusBar().showMessage("Simulation running...")

            # Start simulation
            self.sim_thread.start()

        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}", exc_info=True)
            self.debug_console.append_message(f"Error starting simulation: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to start simulation: {str(e)}")

    def simulation_completed(self, results):
        """Handle simulation completion"""
        # Store results
        self.last_results = results

        # Enable export button
        self.export_data_btn.setEnabled(True)

        # Plot results
        self.plot_results(results)

        # Update signal information
        self.update_signal_info(results)

        # Update UI
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Simulation complete")
        self.debug_console.append_message("Simulation completed successfully")

    def simulation_error(self, error_msg):
        """Handle simulation errors"""
        self.debug_console.append_message(f"Simulation failed: {error_msg}", level='ERROR')
        self.statusBar().showMessage("Simulation failed")
        self.run_btn.setEnabled(True)

        QMessageBox.critical(self, "Simulation Error", f"Simulation failed: {error_msg}")

    def plot_results(self, results):
        """Plot simulation results"""
        try:
            # Clear the canvas
            self.canvas.axes.clear()

            time_points = results['time_points']
            raw_signal = results['raw_signal']
            signal = results['signal']
            simulate_df_f0 = results['config'].get('simulate_df_f0', False)

            if simulate_df_f0 and 'df_f0_signal' in results:
                # Plot DF/F0 signal
                plot_signal = results['df_f0_signal']
                df_f0_as_percent = results['config'].get('df_f0_as_percent', True)
                y_label = 'ΔF/F₀ (%)' if df_f0_as_percent else 'ΔF/F₀ (ratio)'

                # Calculate DF/F0 for the raw signal too (for the ideal trace)
                baseline_end = int(results['config']['agonist_addition_time'] / results['config']['time_interval'])
                if baseline_end > 0 and baseline_end < len(raw_signal):
                    f0 = np.mean(raw_signal[:baseline_end])
                    if df_f0_as_percent:
                        raw_df_f0 = ((raw_signal - f0) / f0) * 100  # As percentage
                    else:
                        raw_df_f0 = (raw_signal - f0) / f0  # As ratio
                    # Plot raw DF/F0 signal
                    self.canvas.axes.plot(time_points, raw_df_f0, 'g--', alpha=0.5, label='Ideal Signal')

                # Plot full DF/F0 signal with noise
                self.canvas.axes.plot(time_points, plot_signal, 'b-', label='Simulated Signal')
            else:
                # Plot original raw signals
                self.canvas.axes.plot(time_points, raw_signal, 'g--', alpha=0.5, label='Ideal Signal')
                self.canvas.axes.plot(time_points, signal, 'b-', label='Simulated Signal')
                y_label = 'Fluorescence (A.U.)'

            # Mark agonist addition time
            agonist_time = results['config']['agonist_addition_time']
            self.canvas.axes.axvline(x=agonist_time, color='r', linestyle='--',
                                   label=f'Agonist Addition ({agonist_time}s)')

            # Calculate and mark baseline
            baseline_end = int(agonist_time / results['config']['time_interval'])
            if baseline_end > 0:
                if simulate_df_f0 and 'df_f0_signal' in results:
                    baseline = 0  # Baseline for DF/F0 is zero by definition
                else:
                    baseline = np.mean(signal[:baseline_end])

                self.canvas.axes.axhline(y=baseline, color='k', linestyle=':',
                                       label=f'Baseline ({baseline:.1f})')

            # Set labels and title
            agonist_type = results['config']['agonist_type']
            concentration = results['config']['concentration']

            self.canvas.axes.set_title(f'{agonist_type} Response ({concentration} μM)')
            self.canvas.axes.set_xlabel('Time (s)')
            self.canvas.axes.set_ylabel(y_label)
            self.canvas.axes.legend()

            # Refresh canvas
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}", exc_info=True)
            self.debug_console.append_message(f"Error plotting results: {str(e)}", level='ERROR')



    def update_signal_info(self, results):
        """Update the signal information display"""
        try:
            # Extract parameters for display
            config = results['config']
            time_points = results['time_points']
            signal = results['signal']
            simulate_df_f0 = config.get('simulate_df_f0', False)

            # Calculate statistics
            baseline_end = int(config['agonist_addition_time'] / config['time_interval'])
            if baseline_end > 0:
                # Calculate appropriate statistics based on signal type
                if simulate_df_f0 and 'df_f0_signal' in results:
                    df_f0_signal = results['df_f0_signal']
                    df_f0_as_percent = config.get('df_f0_as_percent', True)
                    baseline = 0  # DF/F0 baseline is 0 by definition
                    baseline_std = np.std(df_f0_signal[:baseline_end])

                    # Find peak after agonist addition
                    post_addition = df_f0_signal[baseline_end:]
                    if len(post_addition) > 0:
                        peak_value = np.max(post_addition)
                        peak_idx = baseline_end + np.argmax(post_addition)
                        peak_time = time_points[peak_idx]

                        # Amplitude is the peak value itself for DF/F0
                        amplitude = peak_value

                        signal_type = "ΔF/F₀ (%)" if df_f0_as_percent else "ΔF/F₀ (ratio)"

                else:
                    # Original raw fluorescence calculations
                    baseline = np.mean(signal[:baseline_end])
                    baseline_std = np.std(signal[:baseline_end])

                    # Find peak after agonist addition
                    post_addition = signal[baseline_end:]
                    if len(post_addition) > 0:
                        peak_value = np.max(post_addition)
                        peak_idx = baseline_end + np.argmax(post_addition)
                        peak_time = time_points[peak_idx]

                        # Calculate response amplitude
                        amplitude = peak_value - baseline

                        signal_type = "Fluorescence (A.U.)"

                # Calculate dose-response factor
                agonist_type = config['agonist_type']
                concentration = config['concentration']
                dose_factor = self.calculate_dose_response_factor(agonist_type, concentration)

                # Create information text
                info_text = (
                    f"<b>Agonist:</b> {agonist_type} ({concentration} μM)<br>"
                    f"<b>Dose-Response Factor:</b> {dose_factor:.3f}<br>"
                    f"<b>Signal Type:</b> {signal_type}<br>"
                    f"<b>Baseline:</b> {baseline:.1f} ± {baseline_std:.1f}<br>"
                    f"<b>Peak Value:</b> {peak_value:.1f} at {peak_time:.1f}s<br>"
                    f"<b>Response Amplitude:</b> {amplitude:.1f}<br>"
                    f"<b>Rise Rate:</b> {config['rise_rate']:.3f}<br>"
                    f"<b>Decay Rate:</b> {config['decay_rate']:.3f}"
                )

                # Add error information if applicable
                if config['apply_errors']:
                    active_errors = []
                    for error_type, error_info in config['error_types'].items():
                        if error_info.get('enabled', False):
                            active_errors.append(error_type.replace('_', ' ').title())

                    if active_errors:
                        info_text += f"<br><b>Active Errors:</b> {', '.join(active_errors)}"

                self.signal_info_label.setText(info_text)
            else:
                self.signal_info_label.setText("Insufficient data to analyze signal.")

        except Exception as e:
            logger.error(f"Error updating signal information: {str(e)}", exc_info=True)
            if hasattr(self, 'debug_console'):
                self.debug_console.append_message(f"Error updating signal info: {str(e)}", level='ERROR')
            self.signal_info_label.setText("Error analyzing signal.")

    def save_configuration(self):
        """Save the current configuration to a file"""
        try:
            # Get current config
            config = self.get_simulation_config()

            # Get filename from dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Configuration",
                os.path.join(os.getcwd(), "configs", "agonist_config.json"),
                "JSON Files (*.json)"
            )

            if filename:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                # Save configuration to file
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=4)

                self.debug_console.append_message(f"Configuration saved to {filename}")
                self.statusBar().showMessage(f"Configuration saved to {filename}")

        except Exception as e:
            self.debug_console.append_message(f"Error saving configuration: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")

    def load_configuration(self):
        """Load configuration from a file"""
        try:
            # Get filename from dialog
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Configuration",
                os.path.join(os.getcwd(), "configs"),
                "JSON Files (*.json)"
            )

            if filename:
                # Load configuration from file
                with open(filename, 'r') as f:
                    config = json.load(f)

                # Update UI with loaded config
                self.update_ui_from_config(config)

                self.debug_console.append_message(f"Configuration loaded from {filename}")
                self.statusBar().showMessage(f"Configuration loaded from {filename}")

        except Exception as e:
            self.debug_console.append_message(f"Error loading configuration: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")

    def update_ui_from_config(self, config):
        """Update UI elements from a loaded configuration"""
        try:
            # Update agonist selection
            agonist_type = config.get('agonist_type', 'ATP')
            index = self.agonist_combo.findText(agonist_type)
            if index >= 0:
                self.agonist_combo.setCurrentIndex(index)

            # Update values
            self.concentration_spin.setValue(config.get('concentration', 100))
            self.baseline_spin.setValue(config.get('baseline', 500))
            self.peak_spin.setValue(config.get('peak', 1000))
            self.rise_rate_spin.setValue(config.get('rise_rate', 0.05))
            self.decay_rate_spin.setValue(config.get('decay_rate', 0.05))
            self.num_timepoints_spin.setValue(config.get('num_timepoints', 450))
            self.time_interval_spin.setValue(config.get('time_interval', 0.4))
            self.agonist_time_spin.setValue(config.get('agonist_addition_time', 10))
            self.read_noise_spin.setValue(config.get('read_noise', 20))
            self.background_spin.setValue(config.get('background', 100))
            self.photobleaching_spin.setValue(config.get('photobleaching_rate', 0.0005))

            # Update error settings
            self.apply_errors_check.setChecked(config.get('apply_errors', False))

            # Update error type parameters
            error_types = config.get('error_types', {})

            # Random Spikes
            spikes = error_types.get('random_spikes', {})
            self.random_spikes_check.setChecked(spikes.get('enabled', False))
            self.spikes_count_spin.setValue(spikes.get('count', 3))
            self.spikes_amplitude_spin.setValue(spikes.get('amplitude', 1000))
            self.spikes_width_spin.setValue(spikes.get('width', 3))

            # Signal Dropouts
            dropouts = error_types.get('signal_dropouts', {})
            self.signal_dropouts_check.setChecked(dropouts.get('enabled', False))
            self.dropouts_count_spin.setValue(dropouts.get('count', 1))
            self.dropouts_length_spin.setValue(dropouts.get('length', 10))
            self.dropouts_factor_spin.setValue(dropouts.get('factor', 0.2))

            # Baseline Drift
            drift = error_types.get('baseline_drift', {})
            self.baseline_drift_check.setChecked(drift.get('enabled', False))
            direction_index = self.drift_direction_combo.findText(drift.get('direction', 'Rising'))
            if direction_index >= 0:
                self.drift_direction_combo.setCurrentIndex(direction_index)
            self.drift_magnitude_spin.setValue(drift.get('magnitude', 500))

            # Oscillating Baseline
            oscillation = error_types.get('oscillating_baseline', {})
            self.oscillating_baseline_check.setChecked(oscillation.get('enabled', False))
            self.oscillation_frequency_spin.setValue(oscillation.get('frequency', 0.05))
            self.oscillation_amplitude_spin.setValue(oscillation.get('amplitude', 300))

            # Delayed Response
            delay = error_types.get('delayed_response', {})
            self.delayed_response_check.setChecked(delay.get('enabled', False))
            self.delay_time_spin.setValue(delay.get('delay_seconds', 5))

            # Incomplete Decay
            incomplete_decay = error_types.get('incomplete_decay', {})
            self.incomplete_decay_check.setChecked(incomplete_decay.get('enabled', False))
            self.elevation_factor_spin.setValue(incomplete_decay.get('elevation_factor', 0.5))

            # Extra Noise
            extra_noise = error_types.get('extra_noise', {})
            self.extra_noise_check.setChecked(extra_noise.get('enabled', False))
            self.noise_std_spin.setValue(extra_noise.get('std', 100))

            # Sudden Jump
            sudden_jump = error_types.get('sudden_jump', {})
            self.sudden_jump_check.setChecked(sudden_jump.get('enabled', False))
            self.jump_position_spin.setValue(sudden_jump.get('position_pct', 0.7))
            self.jump_magnitude_spin.setValue(sudden_jump.get('magnitude', 500))

            # Signal Cutout
            signal_cutout = error_types.get('signal_cutout', {})
            self.signal_cutout_check.setChecked(signal_cutout.get('enabled', False))
            self.cutout_start_spin.setValue(signal_cutout.get('start_pct', 0.3))
            self.cutout_duration_spin.setValue(signal_cutout.get('duration_pct', 0.2))
            self.replace_baseline_check.setChecked(signal_cutout.get('replace_with_baseline', True))

            # Exponential Drift
            exp_drift = error_types.get('exp_drift', {})
            self.exp_drift_check.setChecked(exp_drift.get('enabled', False))
            direction_index = self.exp_direction_combo.findText(exp_drift.get('direction', 'Upward'))
            if direction_index >= 0:
                self.exp_direction_combo.setCurrentIndex(direction_index)
            self.exp_magnitude_spin.setValue(exp_drift.get('magnitude', 1000))
            self.exp_rate_spin.setValue(exp_drift.get('rate', 3))

        except Exception as e:
            self.debug_console.append_message(f"Error updating UI from config: {str(e)}", level='ERROR')

    def export_plot(self):
        """Export the current plot to a file"""
        try:
            # Get filename from dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Plot",
                os.path.join(os.getcwd(), "exports", f"agonist_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
                "Image Files (*.png *.jpg *.pdf *.svg)"
            )

            if filename:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                # Save figure
                self.canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')

                self.debug_console.append_message(f"Plot exported to {filename}")
                self.statusBar().showMessage(f"Plot exported to {filename}")

        except Exception as e:
            self.debug_console.append_message(f"Error exporting plot: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")

    def export_data(self):
        """Export simulation data to CSV"""
        try:
            if not self.last_results:
                QMessageBox.warning(self, "No Data", "No simulation data available to export.")
                return

            # Get filename from dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Data",
                os.path.join(os.getcwd(), "exports",
                            f"agonist_data_{self.last_results['config']['agonist_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                "CSV Files (*.csv)"
            )

            if filename:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                # Create DataFrame for export
                df_data = {
                    'Time': self.last_results['time_points'],
                    'Raw_Signal': self.last_results['raw_signal'],
                    'Signal': self.last_results['signal']
                }

                # Add DF/F0 data if available
                if self.last_results['config'].get('simulate_df_f0', False) and 'df_f0_signal' in self.last_results:
                    df_data['DF_F0_Signal'] = self.last_results['df_f0_signal']

                df = pd.DataFrame(df_data)

                # Export to CSV
                df.to_csv(filename, index=False)

                # Export configuration
                config_filename = filename.replace('.csv', '_config.json')
                with open(config_filename, 'w') as f:
                    json.dump(self.last_results['config'], f, indent=4)

                self.debug_console.append_message(f"Data exported to {filename}")
                self.debug_console.append_message(f"Configuration exported to {config_filename}")
                self.statusBar().showMessage(f"Data exported to {filename}")

        except Exception as e:
            self.debug_console.append_message(f"Error exporting data: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

    def calculate_dose_response_factor(self, agonist_type, concentration):
        """Calculate the dose-response factor based on agonist type and concentration"""
        # Define EC50 values for different agonists (in µM)
        ec50_values = {
            "ATP": 100.0,
            "UTP": 150.0,
            "Ionomycin": 0.5,
            "Buffer": float('inf'),  # Will result in near-zero response
            "Custom": 100.0  # Default for custom agonist
        }

        # Define Hill coefficients for different agonists
        hill_coefficients = {
            "ATP": 1.5,
            "UTP": 1.3,
            "Ionomycin": 2.0,
            "Buffer": 1.0,
            "Custom": 1.5  # Default for custom agonist
        }

        # Get EC50 and Hill coefficient for the selected agonist
        ec50 = ec50_values.get(agonist_type, 100.0)
        hill = hill_coefficients.get(agonist_type, 1.5)

        # Special case for Buffer (should give minimal response)
        if agonist_type == "Buffer" or concentration <= 0:
            return 0.05  # Minimal response

        # Calculate response using Hill equation
        response_factor = (concentration ** hill) / (ec50 ** hill + concentration ** hill)

        # Ensure factor is between 0 and 1
        return max(0, min(1, response_factor))

    def on_parameter_changed(self):
        """Handle manual changes to signal parameters"""
        if self.auto_updating:
            return

        # Mark parameters as manually overridden
        self.params_overridden = True

        if hasattr(self, 'debug_console'):
            self.debug_console.append_message("Signal parameters manually adjusted")

    def on_concentration_changed(self, value):
        """Update signal parameters when concentration changes"""
        if self.auto_updating or self.params_overridden:
            return

        # Set auto_updating flag to prevent recursive updates
        self.auto_updating = True

        # Update peak based on concentration
        self.update_peak_from_concentration()

        # Reset auto_updating flag
        self.auto_updating = False

        if hasattr(self, 'debug_console'):
            self.debug_console.append_message(f"Concentration changed to {value} µM")

    def update_peak_from_concentration(self):
        """Update peak height based on agonist type and concentration"""
        agonist_type = self.agonist_combo.currentText()
        concentration = self.concentration_spin.value()

        # Calculate dose-response factor
        dose_factor = self.calculate_dose_response_factor(agonist_type, concentration)

        # Base max peaks for each agonist
        max_peaks = {
            "ATP": 1500,
            "UTP": 1200,
            "Ionomycin": 4000,
            "Buffer": 600,
            "Custom": 1500
        }

        # Get baseline
        baseline = self.baseline_spin.value()

        # Calculate peak based on max peak, baseline, and dose factor
        max_peak = max_peaks.get(agonist_type, 1500)
        peak = baseline + (max_peak - baseline) * dose_factor

        # Set the peak value
        self.peak_spin.setValue(peak)

        if hasattr(self, 'debug_console') and not self.auto_updating:
            self.debug_console.append_message(f"Updated peak to {peak:.1f} based on {agonist_type} at {concentration} µM (dose factor: {dose_factor:.3f})")

    def reset_parameter_override(self):
        """Reset parameter override and update based on current agonist and concentration"""
        self.params_overridden = False
        self.on_agonist_changed(self.agonist_combo.currentText())

        if hasattr(self, 'debug_console'):
            self.debug_console.append_message("Parameters reset to automatic values")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
