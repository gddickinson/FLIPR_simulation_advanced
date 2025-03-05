import sys
import os
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                            QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                            QFileDialog, QMessageBox, QTextEdit, QGroupBox,
                            QFormLayout, QLineEdit, QSplitter, QFrame, QSizePolicy,
                            QTableWidgetItem, QTableWidget, QHeaderView, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QTextCursor, QColor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import json

# Import simulation modules
from simulation_engine import SimulationEngine
from config_manager import ConfigManager
from plate_layout import PlateLayoutEditor
from visualization import PlotManager

# Configure logging
LOG_FOLDER = 'logs'
os.makedirs(LOG_FOLDER, exist_ok=True)
log_filename = os.path.join(LOG_FOLDER, f'flipr_simulator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FLIPR_Simulator')

class SimulationThread(QThread):
    """Thread for running simulations without freezing the GUI"""
    progress_update = pyqtSignal(int)
    simulation_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, simulation_engine, config):
        super().__init__()
        self.simulation_engine = simulation_engine
        self.config = config

    def run(self):
        try:
            logger.info("Starting simulation on separate thread")
            # Update progress as simulation runs
            total_steps = 3  # Example: setup, simulate, post-process

            # Setup simulation
            self.progress_update.emit(1)

            # Run simulation
            simulation_results = self.simulation_engine.simulate(self.config)
            self.progress_update.emit(2)

            # Post-process results
            self.progress_update.emit(3)

            logger.info("Simulation completed successfully")
            self.simulation_complete.emit(simulation_results)

        except Exception as e:
            logger.error(f"Error in simulation thread: {str(e)}", exc_info=True)
            self.error_occurred.emit(str(e))

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

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()

        # Initialize components
        self.config_manager = ConfigManager()
        self.simulation_engine = SimulationEngine()
        self.plot_manager = PlotManager()

        self.init_ui()

        # Log application start
        logger.info("FLIPR Simulator application started")

    def init_ui(self):
        """Initialize the UI components"""
        self.setWindowTitle("FLIPR Calcium Response Simulator")
        self.setGeometry(100, 100, 1200, 800)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create tab widget
        self.tabs = QTabWidget()

        # Add tabs
        self.tabs.addTab(self.create_simulation_tab(), "Simulation")
        self.tabs.addTab(self.create_plate_layout_tab(), "Plate Layout")
        self.tabs.addTab(self.create_error_simulation_tab(), "Error Simulation")
        self.tabs.addTab(self.create_batch_processing_tab(), "Batch Processing")
        self.tabs.addTab(self.create_debug_tab(), "Debug Console")
        self.tabs.addTab(self.create_settings_tab(), "Settings")

        # Add status bar
        self.statusBar().showMessage("Ready")

        # Assemble main layout
        main_layout.addWidget(self.tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Log application start
        logger.info("FLIPR Simulator application started")


    def create_debug_tab(self):
        """Create the debug console tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Create debug console
        self.debug_console = DebugConsole()

        # Add console control buttons
        console_controls = QHBoxLayout()

        clear_btn = QPushButton("Clear Console")
        clear_btn.clicked.connect(self.debug_console.clear)

        log_level_combo = QComboBox()
        log_level_combo.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])

        save_log_btn = QPushButton("Save Log")
        save_log_btn.clicked.connect(self.save_log)

        console_controls.addWidget(QLabel("Log Level:"))
        console_controls.addWidget(log_level_combo)
        console_controls.addWidget(clear_btn)
        console_controls.addWidget(save_log_btn)
        console_controls.addStretch()

        layout.addLayout(console_controls)
        layout.addWidget(self.debug_console)

        tab.setLayout(layout)

        # Add a test message to the console
        self.debug_console.append_message("FLIPR Simulator initialized")

        return tab

    def save_log(self):
        """Save the debug console log to a file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Log",
                os.path.join(os.getcwd(), f"flipr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
                "Text Files (*.txt)"
            )

            if filename:
                with open(filename, 'w') as f:
                    f.write(self.debug_console.toPlainText())

                self.statusBar().showMessage(f"Log saved to {filename}")
                self.debug_console.append_message(f"Log saved to {filename}")
        except Exception as e:
            self.debug_console.append_message(f"Error saving log: {str(e)}", level='ERROR')



    def create_simulation_tab(self):
        """Create the main simulation tab"""
        tab = QWidget()
        layout = QVBoxLayout()  # Change to vertical layout for better space utilization

        # Create a horizontal splitter for control panel and visualization area
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - simulation controls
        control_panel = QWidget()
        control_layout = QVBoxLayout()

        # General settings group
        general_group = QGroupBox("Simulation Parameters")
        general_form = QFormLayout()

        self.plate_type_combo = QComboBox()
        self.plate_type_combo.addItems(["96-well", "384-well"])
        general_form.addRow("Plate Type:", self.plate_type_combo)

        self.num_timepoints_spin = QSpinBox()
        self.num_timepoints_spin.setRange(10, 2000)
        self.num_timepoints_spin.setValue(451)
        general_form.addRow("Number of Timepoints:", self.num_timepoints_spin)

        self.time_interval_spin = QDoubleSpinBox()
        self.time_interval_spin.setRange(0.1, 10.0)
        self.time_interval_spin.setValue(0.4)
        self.time_interval_spin.setSingleStep(0.1)
        general_form.addRow("Time Interval (seconds):", self.time_interval_spin)

        self.agonist_time_spin = QDoubleSpinBox()
        self.agonist_time_spin.setRange(1.0, 100.0)
        self.agonist_time_spin.setValue(10.0)
        general_form.addRow("Agonist Addition Time (seconds):", self.agonist_time_spin)

        general_group.setLayout(general_form)
        control_layout.addWidget(general_group)

        # Noise settings group
        noise_group = QGroupBox("Noise Parameters")
        noise_form = QFormLayout()

        self.read_noise_spin = QDoubleSpinBox()
        self.read_noise_spin.setRange(0, 100)
        self.read_noise_spin.setValue(20)
        noise_form.addRow("Read Noise:", self.read_noise_spin)

        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(0, 500)
        self.background_spin.setValue(100)
        noise_form.addRow("Background:", self.background_spin)

        self.photobleaching_spin = QDoubleSpinBox()
        self.photobleaching_spin.setRange(0, 0.01)
        self.photobleaching_spin.setValue(0.0005)
        self.photobleaching_spin.setDecimals(6)
        self.photobleaching_spin.setSingleStep(0.0001)
        noise_form.addRow("Photobleaching Rate:", self.photobleaching_spin)

        noise_group.setLayout(noise_form)
        control_layout.addWidget(noise_group)

        # Random seed group
        seed_group = QGroupBox("Random Seed")
        seed_layout = QHBoxLayout()

        self.random_seed_check = QCheckBox("Use Random Seed")
        self.random_seed_check.setChecked(True)

        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 9999)
        self.random_seed_spin.setValue(42)
        self.random_seed_spin.setEnabled(False)

        self.random_seed_check.toggled.connect(lambda checked: self.random_seed_spin.setEnabled(not checked))

        seed_layout.addWidget(self.random_seed_check)
        seed_layout.addWidget(self.random_seed_spin)

        seed_group.setLayout(seed_layout)
        control_layout.addWidget(seed_group)

        # Plate layout note
        layout_note = QLabel("Note: Configure cell types and agonists in the Plate Layout tab")
        layout_note.setWordWrap(True)
        layout_note.setStyleSheet("font-style: italic; color: #666;")
        control_layout.addWidget(layout_note)

        # Buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)

        self.save_config_btn = QPushButton("Save Configuration")
        self.save_config_btn.clicked.connect(self.save_configuration)

        self.load_config_btn = QPushButton("Load Configuration")
        self.load_config_btn.clicked.connect(self.load_configuration)

        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.save_config_btn)
        button_layout.addWidget(self.load_config_btn)

        control_layout.addLayout(button_layout)
        control_layout.addStretch()

        control_panel.setLayout(control_layout)

        # Right panel - visualization and well selection
        viz_panel = QWidget()
        viz_layout = QVBoxLayout()

        # Create matplotlib canvas (larger now)
        self.canvas = MatplotlibCanvas(viz_panel, width=8, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, viz_panel)

        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas, 1)  # Give it a stretch factor of 1

        # Plot control buttons
        plot_control_layout = QHBoxLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["All Traces", "By Cell Line", "By Agonist", "Heatmap", "Single Trace"])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot_type)

        self.export_plot_btn = QPushButton("Export Plot")
        self.export_plot_btn.clicked.connect(self.export_plot)

        plot_control_layout.addWidget(QLabel("Plot Type:"))
        plot_control_layout.addWidget(self.plot_type_combo)
        plot_control_layout.addWidget(self.export_plot_btn)
        plot_control_layout.addStretch()

        viz_layout.addLayout(plot_control_layout)

        # Add well selector label
        viz_layout.addWidget(QLabel("Select Individual Well"))

        # Create a grid of well buttons
        well_grid = QGridLayout()
        well_grid.setSpacing(2)  # Compact spacing

        # Store the well buttons for later access
        self.well_buttons = {}

        for row in range(8):  # A-H
            for col in range(12):  # 1-12
                well_id = f"{chr(65 + row)}{col + 1}"
                well_btn = QPushButton(well_id)
                well_btn.setFixedSize(40, 30)  # Small fixed size
                well_btn.setCheckable(True)  # Make it checkable (can be toggled)
                well_btn.clicked.connect(lambda checked, wid=well_id: self.on_well_selected(wid))
                well_grid.addWidget(well_btn, row, col)
                self.well_buttons[well_id] = well_btn

        viz_layout.addLayout(well_grid)

        # Add a label to show the currently selected well info
        self.selected_well_label = QLabel("No well selected")
        viz_layout.addWidget(self.selected_well_label)

        viz_panel.setLayout(viz_layout)

        # Add panels to splitter
        splitter.addWidget(control_panel)
        splitter.addWidget(viz_panel)
        splitter.setSizes([300, 900])  # Give more space to visualization

        layout.addWidget(splitter)
        tab.setLayout(layout)

        return tab

    def get_simulation_config(self):
        """Get simulation configuration from the UI elements and plate layout"""
        try:
            # Extract values from UI elements
            config = {
                'num_timepoints': self.num_timepoints_spin.value(),
                'time_interval': self.time_interval_spin.value(),
                'agonist_addition_time': self.agonist_time_spin.value(),
                'read_noise': self.read_noise_spin.value(),
                'background': self.background_spin.value(),
                'photobleaching_rate': self.photobleaching_spin.value(),
                'plate_type': self.plate_type_combo.currentText()
            }

            # Random seed
            if not self.random_seed_check.isChecked():
                config['random_seed'] = self.random_seed_spin.value()
            else:
                config['random_seed'] = None  # Use different seed each time

            # Get the cell line and agonist definitions from config manager
            config['cell_lines'] = self.config_manager.config.get('cell_lines', {})
            config['agonists'] = self.config_manager.config.get('agonists', {})

            # Get cell lines from plate layout
            if hasattr(self, 'cell_layout_table'):
                cell_layout = []
                for row in range(self.cell_layout_table.rowCount()):
                    row_data = []
                    for col in range(self.cell_layout_table.columnCount()):
                        item = self.cell_layout_table.item(row, col)
                        row_data.append(item.text() if item else "Neurotypical")
                    cell_layout.append(row_data)
                config['cell_line_layout'] = cell_layout

            # Get agonists from plate layout
            if hasattr(self, 'agonist_layout_table'):
                agonist_layout = []
                for row in range(self.agonist_layout_table.rowCount()):
                    row_data = []
                    for col in range(self.agonist_layout_table.columnCount()):
                        item = self.agonist_layout_table.item(row, col)
                        row_data.append(item.text() if item else "Buffer")
                    agonist_layout.append(row_data)
                config['agonist_layout'] = agonist_layout

            # Get concentrations from plate layout
            if hasattr(self, 'concentration_layout_table'):
                concentration_layout = []
                for row in range(self.concentration_layout_table.rowCount()):
                    row_data = []
                    for col in range(self.concentration_layout_table.columnCount()):
                        item = self.concentration_layout_table.item(row, col)
                        try:
                            # Try to convert to float, use 0 if not possible
                            row_data.append(float(item.text()) if item else 0.0)
                        except ValueError:
                            row_data.append(0.0)
                    concentration_layout.append(row_data)
                config['concentration_layout'] = concentration_layout

            # Log configuration
            self.debug_console.append_message(f"Configuration: {config['num_timepoints']} timepoints, "
                                             f"{config['time_interval']}s interval, "
                                             f"agonist at {config['agonist_addition_time']}s")

            return config

        except Exception as e:
            self.debug_console.append_message(f"Error getting simulation config", level='ERROR')
            logger.error(f"Error getting simulation config: {str(e)}", exc_info=True)


    def update_plot_type(self, plot_type):
        """Update the plot based on selected type"""
        if hasattr(self, 'last_results'):
            self.debug_console.append_message(f"Updating plot type to: {plot_type}")

            # Call appropriate plot method based on type
            if plot_type == "All Traces":
                self.plot_results(self.last_results)
            elif plot_type == "By Cell Line":
                self.plot_by_cell_line(self.last_results)
            elif plot_type == "By Agonist":
                self.plot_by_agonist(self.last_results)
            elif plot_type == "Heatmap":
                self.plot_heatmap(self.last_results)
            elif plot_type == "Single Trace":
                self.show_individual_trace()

    def plot_by_cell_line(self, results):
        """Plot results grouped by cell line, separated by agonist type"""
        try:
            # Clear the canvas
            self.canvas.axes.clear()

            if not results or 'plate_data' not in results or 'metadata' not in results:
                self.canvas.axes.text(0.5, 0.5, 'No valid data for cell line plot',
                                   ha='center', va='center', transform=self.canvas.axes.transAxes)
                self.canvas.draw()
                return

            # Group data by cell line AND agonist type
            grouped_data = {}
            time_points = results['time_points']

            for i, well_meta in enumerate(results['metadata']):
                if well_meta.get('valid', True):
                    cell_line = well_meta.get('cell_line', 'Unknown')
                    agonist = well_meta.get('agonist', 'Unknown')

                    # Create a key combining cell line and agonist
                    group_key = f"{cell_line}"

                    if group_key not in grouped_data:
                        grouped_data[group_key] = {}

                    if agonist not in grouped_data[group_key]:
                        grouped_data[group_key][agonist] = []

                    grouped_data[group_key][agonist].append(results['plate_data'][i])

            # Colors for cell lines and line styles for agonists
            cell_line_colors = {
                'Neurotypical': '#4DAF4A',  # Green
                'ASD': '#984EA3',           # Purple
                'FXS': '#FF7F00',           # Orange
                'Unknown': '#999999'        # Gray
            }

            agonist_styles = {
                'ATP': '-',          # Solid line
                'Ionomycin': '--',   # Dashed line
                'Buffer': ':',       # Dotted line
                'Unknown': '-.'      # Dash-dot line
            }

            # Plot each cell line with different agonists as different line styles
            for cell_line, agonists_data in grouped_data.items():
                color = cell_line_colors.get(cell_line, cell_line_colors['Unknown'])

                for agonist, traces in agonists_data.items():
                    if len(traces) == 0:
                        continue

                    linestyle = agonist_styles.get(agonist, agonist_styles['Unknown'])
                    traces_array = np.array(traces)
                    mean_trace = np.mean(traces_array, axis=0)
                    std_trace = np.std(traces_array, axis=0)

                    # Label with both cell line and agonist
                    label = f"{cell_line} - {agonist}"

                    self.canvas.axes.plot(time_points, mean_trace, label=label,
                                       color=color, linestyle=linestyle, linewidth=2)
                    self.canvas.axes.fill_between(time_points,
                                               mean_trace - std_trace,
                                               mean_trace + std_trace,
                                               color=color, alpha=0.2)

            # Set labels and title
            self.canvas.axes.set_title('Calcium Response by Cell Line and Agonist')
            self.canvas.axes.set_xlabel('Time (s)')
            self.canvas.axes.set_ylabel('Fluorescence (A.U.)')

            # Add legend with smaller font to accommodate more entries
            self.canvas.axes.legend(fontsize='small', loc='upper right')

            # Mark agonist addition time if available
            if 'params' in results and 'agonist_addition_time' in results['params']:
                agonist_time = results['params']['agonist_addition_time']
                self.canvas.axes.axvline(x=agonist_time, color='black', linestyle='--',
                                      label=f'Agonist ({agonist_time}s)')

            # Refresh canvas
            self.canvas.draw()

        except Exception as e:
            self.debug_console.append_message(f"Error plotting by cell line: {str(e)}", level='ERROR')
            logger.error(f"Error plotting by cell line: {str(e)}", exc_info=True)

    def plot_by_agonist(self, results):
        """Plot results grouped by agonist, separated by cell type"""
        try:
            # Clear the canvas
            self.canvas.axes.clear()

            if not results or 'plate_data' not in results or 'metadata' not in results:
                self.canvas.axes.text(0.5, 0.5, 'No valid data for agonist plot',
                                   ha='center', va='center', transform=self.canvas.axes.transAxes)
                self.canvas.draw()
                return

            # Group data by agonist AND cell line
            grouped_data = {}
            time_points = results['time_points']

            for i, well_meta in enumerate(results['metadata']):
                if well_meta.get('valid', True):
                    cell_line = well_meta.get('cell_line', 'Unknown')
                    agonist = well_meta.get('agonist', 'Unknown')
                    concentration = well_meta.get('concentration', 0)

                    # Create a key combining agonist and concentration
                    group_key = f"{agonist} ({concentration} µM)"

                    if group_key not in grouped_data:
                        grouped_data[group_key] = {}

                    if cell_line not in grouped_data[group_key]:
                        grouped_data[group_key][cell_line] = []

                    grouped_data[group_key][cell_line].append(results['plate_data'][i])

            # Colors for agonists
            agonist_colors = {
                'ATP': '#66C2A5',       # Teal
                'UTP': '#FC8D62',       # Orange
                'Ionomycin': '#E78AC3',  # Pink
                'Buffer': '#A6D854',    # Green
                'Unknown': '#999999'    # Gray
            }

            # Line styles for cell lines
            cell_line_styles = {
                'Neurotypical': '-',    # Solid line
                'ASD': '--',            # Dashed line
                'FXS': ':',             # Dotted line
                'Unknown': '-.'         # Dash-dot line
            }

            # Plot each agonist with different cell lines as different line styles
            for agonist_key, cell_lines_data in grouped_data.items():
                # Extract base agonist name for color lookup
                base_agonist = agonist_key.split(' ')[0]
                color = agonist_colors.get(base_agonist, agonist_colors['Unknown'])

                for cell_line, traces in cell_lines_data.items():
                    if len(traces) == 0:
                        continue

                    linestyle = cell_line_styles.get(cell_line, cell_line_styles['Unknown'])
                    traces_array = np.array(traces)
                    mean_trace = np.mean(traces_array, axis=0)
                    std_trace = np.std(traces_array, axis=0)

                    # Label with both agonist and cell line
                    label = f"{agonist_key} - {cell_line}"

                    self.canvas.axes.plot(time_points, mean_trace, label=label,
                                       color=color, linestyle=linestyle, linewidth=2)
                    self.canvas.axes.fill_between(time_points,
                                               mean_trace - std_trace,
                                               mean_trace + std_trace,
                                               color=color, alpha=0.2)

            # Set labels and title
            self.canvas.axes.set_title('Calcium Response by Agonist and Cell Line')
            self.canvas.axes.set_xlabel('Time (s)')
            self.canvas.axes.set_ylabel('Fluorescence (A.U.)')

            # Add legend with smaller font to accommodate more entries
            self.canvas.axes.legend(fontsize='small', loc='upper right')

            # Mark agonist addition time if available
            if 'params' in results and 'agonist_addition_time' in results['params']:
                agonist_time = results['params']['agonist_addition_time']
                self.canvas.axes.axvline(x=agonist_time, color='black', linestyle='--')

            # Refresh canvas
            self.canvas.draw()

        except Exception as e:
            self.debug_console.append_message(f"Error plotting by agonist: {str(e)}", level='ERROR')
            logger.error(f"Error plotting by agonist: {str(e)}", exc_info=True)

    def plot_heatmap(self, results):
        """Plot results as a plate heatmap"""
        try:
            # Clear the canvas
            self.canvas.axes.clear()

            if not results or 'plate_data' not in results or 'params' not in results:
                self.canvas.axes.text(0.5, 0.5, 'No valid data for heatmap plot',
                                   ha='center', va='center', transform=self.canvas.axes.transAxes)
                self.canvas.draw()
                return

            # Get plate dimensions
            if results['params'].get('plate_type', '96-well') == '96-well':
                rows, cols = 8, 12
            else:
                rows, cols = 16, 24

            # Calculate peak responses
            peak_data = np.zeros((rows, cols))
            baseline_end = int(results['params'].get('agonist_addition_time', 10) /
                             results['params'].get('time_interval', 0.4))

            for i, well_data in enumerate(results['plate_data']):
                if i < rows * cols:
                    row, col = i // cols, i % cols
                    if len(well_data) > baseline_end:
                        baseline = np.mean(well_data[:baseline_end]) if baseline_end > 0 else well_data[0]
                        peak = np.max(well_data[baseline_end:])
                        peak_data[row, col] = peak - baseline

            # Create heatmap
            im = self.canvas.axes.imshow(peak_data, cmap='viridis')
            plt.colorbar(im, ax=self.canvas.axes, label='Peak Response (F-F0)')

            # Add well labels
            for i in range(rows):
                for j in range(cols):
                    self.canvas.axes.text(j, i, f"{chr(65+i)}{j+1}", ha='center', va='center',
                                      color='white', fontsize=8)

            # Set title and labels
            self.canvas.axes.set_title('Peak Response Heatmap')

            # Refresh canvas
            self.canvas.draw()

        except Exception as e:
            self.debug_console.append_message(f"Error plotting heatmap: {str(e)}", level='ERROR')
            logger.error(f"Error plotting heatmap: {str(e)}", exc_info=True)

    def simulation_completed(self, results):
        """Handle simulation completion"""
        self.debug_console.append_message("Simulation completed successfully")
        self.statusBar().showMessage("Simulation complete")

        # Store results for later use
        self.last_results = results

        # Plot results based on current plot type
        plot_type = self.plot_type_combo.currentText()
        if plot_type == "All Traces":
            self.plot_results(results)
        elif plot_type == "By Cell Line":
            self.plot_by_cell_line(results)
        elif plot_type == "By Agonist":
            self.plot_by_agonist(results)
        elif plot_type == "Heatmap":
            self.plot_heatmap(results)

    def save_configuration(self):
        """Save current configuration"""
        try:
            # Get current config
            config = self.get_simulation_config()

            # Get filename
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Configuration",
                os.path.join(self.config_manager.config_dir, "user_config.json"),
                "JSON Files (*.json)"
            )

            if filename:
                self.config_manager.save_config(filename, config)
                self.debug_console.append_message(f"Configuration saved to {filename}")
                self.statusBar().showMessage(f"Configuration saved to {filename}")

        except Exception as e:
            self.debug_console.append_message(f"Error saving configuration: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")

    def load_configuration(self):
        """Load configuration from file"""
        try:
            # Get filename
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Configuration",
                self.config_manager.config_dir,
                "JSON Files (*.json)"
            )

            if filename:
                config = self.config_manager.load_config(filename)

                if config:
                    # Update UI elements with loaded values
                    if 'num_timepoints' in config:
                        self.num_timepoints_spin.setValue(config['num_timepoints'])

                    if 'time_interval' in config:
                        self.time_interval_spin.setValue(config['time_interval'])

                    if 'agonist_addition_time' in config:
                        self.agonist_time_spin.setValue(config['agonist_addition_time'])

                    if 'plate_type' in config:
                        index = self.plate_type_combo.findText(config['plate_type'])
                        if index >= 0:
                            self.plate_type_combo.setCurrentIndex(index)

                    self.debug_console.append_message(f"Configuration loaded from {filename}")
                    self.statusBar().showMessage(f"Configuration loaded from {filename}")

        except Exception as e:
            self.debug_console.append_message(f"Error loading configuration: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")

    def export_plot(self):
        """Export current plot to file"""
        try:
            # Get filename
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Plot",
                os.path.join(os.getcwd(), "plot.png"),
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )

            if filename:
                self.canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.debug_console.append_message(f"Plot exported to {filename}")
                self.statusBar().showMessage(f"Plot exported to {filename}")

        except Exception as e:
            self.debug_console.append_message(f"Error exporting plot: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")



    def create_plate_layout_tab(self):
        """Create the plate layout editor tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Top section with controls
        controls_layout = QHBoxLayout()

        # Plate format selector
        format_layout = QVBoxLayout()
        format_layout.addWidget(QLabel("Plate Format:"))

        self.plate_format_combo = QComboBox()
        self.plate_format_combo.addItems(["96-well", "384-well"])
        self.plate_format_combo.currentTextChanged.connect(self.on_plate_format_changed)
        format_layout.addWidget(self.plate_format_combo)
        controls_layout.addLayout(format_layout)

        # Add cell line controls group
        cell_control_group = QGroupBox("Cell Lines")
        cell_control_layout = QVBoxLayout()

        # Cell line by column
        cell_col_layout = QHBoxLayout()
        cell_col_layout.addWidget(QLabel("Column:"))

        self.cell_col_combo = QComboBox()
        self.cell_col_combo.addItems([str(i) for i in range(1, 13)])  # 1-12 for 96-well
        cell_col_layout.addWidget(self.cell_col_combo)

        cell_col_layout.addWidget(QLabel("Cell Type:"))

        self.cell_type_combo = QComboBox()
        self.cell_type_combo.addItems(["Neurotypical", "ASD", "FXS"])
        cell_col_layout.addWidget(self.cell_type_combo)

        self.apply_cell_btn = QPushButton("Apply to Column")
        self.apply_cell_btn.clicked.connect(self.apply_cell_to_column)
        cell_col_layout.addWidget(self.apply_cell_btn)

        cell_control_layout.addLayout(cell_col_layout)

        # Cell line concentrations
        cell_control_layout.addWidget(QLabel("Cell Characteristics:"))

        self.cell_property_table = QTableWidget(3, 5)
        self.cell_property_table.setHorizontalHeaderLabels(["Cell Type", "Baseline", "Peak (Ionomycin)", "Peak (Other)", "Decay Rate"])
        self.cell_property_table.verticalHeader().setVisible(False)
        self.cell_property_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        # Fill in default cell properties
        cell_types = ["Neurotypical", "ASD", "FXS"]
        for i, cell_type in enumerate(cell_types):
            self.cell_property_table.setItem(i, 0, QTableWidgetItem(cell_type))
            properties = self.config_manager.config.get('cell_lines', {}).get(cell_type, {})
            self.cell_property_table.setItem(i, 1, QTableWidgetItem(str(properties.get('baseline', 500))))
            self.cell_property_table.setItem(i, 2, QTableWidgetItem(str(properties.get('peak_ionomycin', 0))))
            self.cell_property_table.setItem(i, 3, QTableWidgetItem(str(properties.get('peak_other', 0))))
            self.cell_property_table.setItem(i, 4, QTableWidgetItem(str(properties.get('decay_rate', 0))))

        cell_control_layout.addWidget(self.cell_property_table)

        cell_control_group.setLayout(cell_control_layout)
        controls_layout.addWidget(cell_control_group)

        # Add agonist controls group
        agonist_control_group = QGroupBox("Agonists")
        agonist_control_layout = QVBoxLayout()

        # Agonist by row
        agonist_row_layout = QHBoxLayout()
        agonist_row_layout.addWidget(QLabel("Row:"))

        self.agonist_row_combo = QComboBox()
        self.agonist_row_combo.addItems(["A", "B", "C", "D", "E", "F", "G", "H"])  # A-H for 96-well
        agonist_row_layout.addWidget(self.agonist_row_combo)

        agonist_row_layout.addWidget(QLabel("Agonist:"))

        self.agonist_combo = QComboBox()
        self.agonist_combo.addItems(["ATP", "UTP", "Ionomycin", "Buffer"])
        agonist_row_layout.addWidget(self.agonist_combo)

        # Concentration input
        agonist_row_layout.addWidget(QLabel("Conc:"))

        self.concentration_spin = QDoubleSpinBox()
        self.concentration_spin.setRange(0.001, 1000)
        self.concentration_spin.setValue(100)
        self.concentration_spin.setDecimals(3)
        agonist_row_layout.addWidget(self.concentration_spin)

        self.concentration_unit = QComboBox()
        self.concentration_unit.addItems(["nM", "µM", "mM"])
        self.concentration_unit.setCurrentText("µM")
        agonist_row_layout.addWidget(self.concentration_unit)

        self.apply_agonist_btn = QPushButton("Apply to Row")
        self.apply_agonist_btn.clicked.connect(self.apply_agonist_to_row)
        agonist_row_layout.addWidget(self.apply_agonist_btn)

        agonist_control_layout.addLayout(agonist_row_layout)

        # Agonist properties
        agonist_control_layout.addWidget(QLabel("Agonist Properties:"))

        self.agonist_property_table = QTableWidget(4, 3)
        self.agonist_property_table.setHorizontalHeaderLabels(["Agonist", "Response Factor", "EC50 (µM)"])
        self.agonist_property_table.verticalHeader().setVisible(False)
        self.agonist_property_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        # Fill in default agonist properties
        agonists = ["ATP", "UTP", "Ionomycin", "Buffer"]
        ec50_values = {"ATP": 100.0, "UTP": 150.0, "Ionomycin": 0.5, "Buffer": "N/A"}

        for i, agonist in enumerate(agonists):
            self.agonist_property_table.setItem(i, 0, QTableWidgetItem(agonist))
            agonist_factor = self.config_manager.config.get('agonists', {}).get(agonist, 1.0)
            self.agonist_property_table.setItem(i, 1, QTableWidgetItem(str(agonist_factor)))
            self.agonist_property_table.setItem(i, 2, QTableWidgetItem(str(ec50_values.get(agonist, "N/A"))))

        agonist_control_layout.addWidget(self.agonist_property_table)

        agonist_control_group.setLayout(agonist_control_layout)
        controls_layout.addWidget(agonist_control_group)

        layout.addLayout(controls_layout)

        # Plate visualization
        plate_layout_group = QGroupBox("Plate Layout")
        plate_layout = QVBoxLayout()

        tab_layout = QTabWidget()

        # Create tables for cell line and agonist layouts
        self.cell_layout_table = QTableWidget(8, 12)  # 8 rows x 12 columns for 96-well
        self.cell_layout_table.setHorizontalHeaderLabels([str(i) for i in range(1, 13)])
        self.cell_layout_table.setVerticalHeaderLabels(["A", "B", "C", "D", "E", "F", "G", "H"])

        self.agonist_layout_table = QTableWidget(8, 12)  # 8 rows x 12 columns for 96-well
        self.agonist_layout_table.setHorizontalHeaderLabels([str(i) for i in range(1, 13)])
        self.agonist_layout_table.setVerticalHeaderLabels(["A", "B", "C", "D", "E", "F", "G", "H"])

        self.concentration_layout_table = QTableWidget(8, 12)  # 8 rows x 12 columns for 96-well
        self.concentration_layout_table.setHorizontalHeaderLabels([str(i) for i in range(1, 13)])
        self.concentration_layout_table.setVerticalHeaderLabels(["A", "B", "C", "D", "E", "F", "G", "H"])

        # Set up cell sizes
        for table in [self.cell_layout_table, self.agonist_layout_table, self.concentration_layout_table]:
            for i in range(8):
                table.setRowHeight(i, 40)
            for i in range(12):
                table.setColumnWidth(i, 80)

        # Initialize tables with default values
        self.initialize_layout_tables()

        # Add tables to tabs
        tab_layout.addTab(self.cell_layout_table, "Cell Lines")
        tab_layout.addTab(self.agonist_layout_table, "Agonists")
        tab_layout.addTab(self.concentration_layout_table, "Concentrations (µM)")

        plate_layout.addWidget(tab_layout)

        # Add buttons for saving/loading layouts
        button_layout = QHBoxLayout()

        self.save_layout_btn = QPushButton("Save Layout")
        self.save_layout_btn.clicked.connect(self.save_plate_layout)

        self.load_layout_btn = QPushButton("Load Layout")
        self.load_layout_btn.clicked.connect(self.load_plate_layout)

        self.reset_layout_btn = QPushButton("Reset to Default")
        self.reset_layout_btn.clicked.connect(self.reset_plate_layout)

        button_layout.addWidget(self.save_layout_btn)
        button_layout.addWidget(self.load_layout_btn)
        button_layout.addWidget(self.reset_layout_btn)
        button_layout.addStretch()

        plate_layout.addLayout(button_layout)

        plate_layout_group.setLayout(plate_layout)
        layout.addWidget(plate_layout_group)

        tab.setLayout(layout)

        return tab

    def initialize_layout_tables(self):
        """Initialize the layout tables with default values"""
        # Initialize cell layout with specific pattern
        # Column-based layout: (ASD, FXS, Neurotypical, ASD, ASD, Neurotypical, FXS, Neurotypical, Neurotypical, ASD, ASD, Neurotypical)
        cell_column_types = [
            "ASD", "FXS", "Neurotypical", "ASD", "ASD", "Neurotypical",
            "FXS", "Neurotypical", "Neurotypical", "ASD", "ASD", "Neurotypical"
        ]

        for row in range(8):
            for col in range(12):
                cell_type = cell_column_types[col]
                self.cell_layout_table.setItem(row, col, QTableWidgetItem(cell_type))

                # Set background color based on cell type
                self._set_cell_background_color(self.cell_layout_table, row, col, cell_type)

        # Initialize agonist layout
        # First 3 rows (A-C): ATP
        # Next 3 rows (D-F): Ionomycin
        # Last 2 rows (G-H): Buffer
        for row in range(8):
            for col in range(12):
                if row < 3:
                    agonist = "ATP"
                elif row < 6:
                    agonist = "Ionomycin"
                else:
                    agonist = "Buffer"

                self.agonist_layout_table.setItem(row, col, QTableWidgetItem(agonist))

                # Set background color based on agonist
                self._set_agonist_background_color(self.agonist_layout_table, row, col, agonist)

        # Initialize concentration layout
        for row in range(8):
            for col in range(12):
                if row < 3:  # ATP rows
                    conc = "100"
                elif row < 6:  # Ionomycin rows
                    conc = "1"
                else:  # Buffer rows
                    conc = "0"

                self.concentration_layout_table.setItem(row, col, QTableWidgetItem(conc))

    def _set_cell_background_color(self, table, row, col, cell_type):
        """Set cell background color based on cell type"""
        colors = {
            "Neurotypical": QColor("#4DAF4A"),  # Green
            "ASD": QColor("#984EA3"),           # Purple
            "FXS": QColor("#FF7F00")            # Orange
        }

        if cell_type in colors:
            item = table.item(row, col)
            if item:
                color = colors[cell_type]
                # Make the color semi-transparent
                color.setAlpha(100)
                item.setBackground(color)

    def _set_agonist_background_color(self, table, row, col, agonist):
        """Set cell background color based on agonist"""
        colors = {
            "ATP": QColor("#66C2A5"),       # Teal
            "UTP": QColor("#FC8D62"),       # Orange
            "Ionomycin": QColor("#E78AC3"),  # Pink
            "Buffer": QColor("#A6D854")      # Green
        }

        if agonist in colors:
            item = table.item(row, col)
            if item:
                color = colors[agonist]
                # Make the color semi-transparent
                color.setAlpha(100)
                item.setBackground(color)

    def on_plate_format_changed(self, format_text):
        """Handle changes to plate format"""
        # Update rows/columns based on plate format
        if format_text == "96-well":
            rows, cols = 8, 12
        else:  # 384-well
            rows, cols = 16, 24

        # Reinitialize tables with the new format
        # This would need more implementation for 384-well support

        # Update UI elements
        if format_text == "384-well":
            self.debug_console.append_message("384-well format not fully implemented", level="WARNING")

    def apply_cell_to_column(self):
        """Apply selected cell type to an entire column"""
        try:
            col = int(self.cell_col_combo.currentText()) - 1  # Convert to 0-based index
            cell_type = self.cell_type_combo.currentText()

            # Update the cell layout table
            for row in range(8):  # Assuming 96-well plate
                self.cell_layout_table.setItem(row, col, QTableWidgetItem(cell_type))
                self._set_cell_background_color(self.cell_layout_table, row, col, cell_type)

            self.debug_console.append_message(f"Applied cell type {cell_type} to column {col+1}")
        except Exception as e:
            self.debug_console.append_message(f"Error applying cell type: {str(e)}", level="ERROR")

    def apply_agonist_to_row(self):
        """Apply selected agonist to an entire row"""
        try:
            row_letter = self.agonist_row_combo.currentText()
            row = ord(row_letter) - ord('A')  # Convert letter to 0-based index
            agonist = self.agonist_combo.currentText()

            # Get concentration
            concentration = self.concentration_spin.value()
            unit = self.concentration_unit.currentText()

            # Convert to µM for display
            if unit == "nM":
                display_conc = str(concentration / 1000)
            elif unit == "mM":
                display_conc = str(concentration * 1000)
            else:  # µM
                display_conc = str(concentration)

            # Update the agonist layout table
            for col in range(12):  # Assuming 96-well plate
                self.agonist_layout_table.setItem(row, col, QTableWidgetItem(agonist))
                self._set_agonist_background_color(self.agonist_layout_table, row, col, agonist)

                # Update concentration table
                self.concentration_layout_table.setItem(row, col, QTableWidgetItem(display_conc))

            self.debug_console.append_message(f"Applied agonist {agonist} ({concentration} {unit}) to row {row_letter}")
        except Exception as e:
            self.debug_console.append_message(f"Error applying agonist: {str(e)}", level="ERROR")

    def save_plate_layout(self):
        """Save the current plate layout"""
        try:
            # Create a layout object to save
            layout = {
                'plate_format': self.plate_format_combo.currentText(),
                'cell_lines': self._get_table_data(self.cell_layout_table),
                'agonists': self._get_table_data(self.agonist_layout_table),
                'concentrations': self._get_table_data(self.concentration_layout_table)
            }

            # Get filename
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Plate Layout",
                os.path.join(self.config_manager.config_dir, "plate_layout.json"),
                "JSON Files (*.json)"
            )

            if filename:
                with open(filename, 'w') as f:
                    json.dump(layout, f, indent=4)

                self.debug_console.append_message(f"Plate layout saved to {filename}")
                self.statusBar().showMessage(f"Plate layout saved to {filename}")
        except Exception as e:
            self.debug_console.append_message(f"Error saving plate layout: {str(e)}", level="ERROR")
            QMessageBox.critical(self, "Error", f"Failed to save plate layout: {str(e)}")

    def load_plate_layout(self):
        """Load a plate layout from file"""
        try:
            # Get filename
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Plate Layout",
                self.config_manager.config_dir,
                "JSON Files (*.json)"
            )

            if filename:
                with open(filename, 'r') as f:
                    layout = json.load(f)

                # Update plate format
                if 'plate_format' in layout:
                    self.plate_format_combo.setCurrentText(layout['plate_format'])

                # Update tables
                if 'cell_lines' in layout:
                    self._set_table_data(self.cell_layout_table, layout['cell_lines'])
                    # Update colors
                    for row in range(self.cell_layout_table.rowCount()):
                        for col in range(self.cell_layout_table.columnCount()):
                            item = self.cell_layout_table.item(row, col)
                            if item:
                                self._set_cell_background_color(self.cell_layout_table, row, col, item.text())

                if 'agonists' in layout:
                    self._set_table_data(self.agonist_layout_table, layout['agonists'])
                    # Update colors
                    for row in range(self.agonist_layout_table.rowCount()):
                        for col in range(self.agonist_layout_table.columnCount()):
                            item = self.agonist_layout_table.item(row, col)
                            if item:
                                self._set_agonist_background_color(self.agonist_layout_table, row, col, item.text())

                if 'concentrations' in layout:
                    self._set_table_data(self.concentration_layout_table, layout['concentrations'])

                self.debug_console.append_message(f"Plate layout loaded from {filename}")
                self.statusBar().showMessage(f"Plate layout loaded from {filename}")
        except Exception as e:
            self.debug_console.append_message(f"Error loading plate layout: {str(e)}", level="ERROR")
            QMessageBox.critical(self, "Error", f"Failed to load plate layout: {str(e)}")

    def reset_plate_layout(self):
        """Reset to default plate layout"""
        try:
            # Confirm reset
            reply = QMessageBox.question(self, "Reset Plate Layout",
                                        "Are you sure you want to reset the plate layout to default?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.initialize_layout_tables()
                self.debug_console.append_message("Plate layout reset to default")
        except Exception as e:
            self.debug_console.append_message(f"Error resetting plate layout: {str(e)}", level="ERROR")

    def _get_table_data(self, table):
        """Get all data from a table as a 2D list"""
        data = []
        for row in range(table.rowCount()):
            row_data = []
            for col in range(table.columnCount()):
                item = table.item(row, col)
                row_data.append(item.text() if item else "")
            data.append(row_data)
        return data

    def _set_table_data(self, table, data):
        """Set all data in a table from a 2D list"""
        for row in range(min(len(data), table.rowCount())):
            for col in range(min(len(data[row]), table.columnCount())):
                table.setItem(row, col, QTableWidgetItem(str(data[row][col])))

    def create_error_simulation_tab(self):
        """Create the error simulation tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Create groups for different error types

        # Cell-based errors
        cell_error_group = QGroupBox("Cell-Based Errors")
        cell_error_layout = QVBoxLayout()

        cell_variability = QCheckBox("Cell Variability")
        cell_loading = QCheckBox("Dye Loading Issues")
        cell_health = QCheckBox("Cell Health Problems")
        cell_density = QCheckBox("Variable Cell Density")

        cell_error_layout.addWidget(cell_variability)
        cell_error_layout.addWidget(cell_loading)
        cell_error_layout.addWidget(cell_health)
        cell_error_layout.addWidget(cell_density)

        cell_error_group.setLayout(cell_error_layout)

        # Reagent errors
        reagent_error_group = QGroupBox("Reagent-Based Errors")
        reagent_error_layout = QVBoxLayout()

        reagent_stability = QCheckBox("Agonist Stability Issues")
        reagent_concentration = QCheckBox("Incorrect Concentrations")
        reagent_contamination = QCheckBox("Reagent Contamination")

        reagent_error_layout.addWidget(reagent_stability)
        reagent_error_layout.addWidget(reagent_concentration)
        reagent_error_layout.addWidget(reagent_contamination)

        reagent_error_group.setLayout(reagent_error_layout)

        # Equipment errors
        equipment_error_group = QGroupBox("Equipment-Based Errors")
        equipment_error_layout = QVBoxLayout()

        camera_error = QCheckBox("Camera Errors")
        liquid_handler = QCheckBox("Liquid Handler Issues")
        timing_error = QCheckBox("Timing Inconsistencies")
        focus_error = QCheckBox("Focus Problems")

        equipment_error_layout.addWidget(camera_error)
        equipment_error_layout.addWidget(liquid_handler)
        equipment_error_layout.addWidget(timing_error)
        equipment_error_layout.addWidget(focus_error)

        equipment_error_group.setLayout(equipment_error_layout)

        # Systematic errors
        systematic_error_group = QGroupBox("Systematic Errors")
        systematic_error_layout = QVBoxLayout()

        edge_effect = QCheckBox("Plate Edge Effects")
        temperature = QCheckBox("Temperature Gradients")
        evaporation = QCheckBox("Evaporation")
        well_crosstalk = QCheckBox("Well-to-Well Crosstalk")

        systematic_error_layout.addWidget(edge_effect)
        systematic_error_layout.addWidget(temperature)
        systematic_error_layout.addWidget(evaporation)
        systematic_error_layout.addWidget(well_crosstalk)

        systematic_error_group.setLayout(systematic_error_layout)

        # Create error probability slider
        error_prob_layout = QFormLayout()
        error_prob_spin = QDoubleSpinBox()
        error_prob_spin.setRange(0, 1)
        error_prob_spin.setSingleStep(0.05)
        error_prob_spin.setValue(0.1)
        error_prob_layout.addRow("Error Probability:", error_prob_spin)

        # Assemble layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(cell_error_group)
        top_layout.addWidget(reagent_error_group)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(equipment_error_group)
        bottom_layout.addWidget(systematic_error_group)

        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)
        layout.addLayout(error_prob_layout)

        tab.setLayout(layout)

        return tab

    def create_batch_processing_tab(self):
        """Create the batch processing tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # TODO: Implement batch processing interface
        placeholder = QLabel("Batch Processing will be implemented here")
        placeholder.setAlignment(Qt.AlignCenter)

        layout.addWidget(placeholder)
        tab.setLayout(layout)

        return tab

    def create_settings_tab(self):
        """Create the settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Application settings form
        settings_form = QFormLayout()

        output_dir = QLineEdit()
        output_dir_btn = QPushButton("Browse...")

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(output_dir)
        output_dir_layout.addWidget(output_dir_btn)

        settings_form.addRow("Output Directory:", output_dir_layout)

        default_format_combo = QComboBox()
        default_format_combo.addItems(["CSV", "Excel"])
        settings_form.addRow("Default Output Format:", default_format_combo)

        auto_save = QCheckBox("Auto-save Results")
        settings_form.addRow("", auto_save)

        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout()

        num_threads_spin = QSpinBox()
        num_threads_spin.setRange(1, 16)
        num_threads_spin.setValue(4)
        advanced_layout.addRow("Number of Threads:", num_threads_spin)

        random_seed = QSpinBox()
        random_seed.setRange(0, 9999)
        random_seed.setValue(42)
        random_seed_layout = QHBoxLayout()
        random_seed_layout.addWidget(random_seed)
        random_seed_check = QCheckBox("Use Random Seed")
        random_seed_check.setChecked(True)
        random_seed_layout.addWidget(random_seed_check)

        advanced_layout.addRow("Random Seed:", random_seed_layout)

        advanced_group.setLayout(advanced_layout)

        # Assemble layout
        layout.addLayout(settings_form)
        layout.addWidget(advanced_group)
        layout.addStretch()

        # Save button
        save_btn = QPushButton("Save Settings")
        layout.addWidget(save_btn)

        tab.setLayout(layout)

        return tab

    def run_simulation(self):
        """Run a simulation with the current configuration"""
        try:
            # Get configuration from UI
            config = self.get_simulation_config()

            # Log simulation start
            self.debug_console.append_message(f"Starting simulation with {config['num_timepoints']} timepoints")

            # Create and start simulation thread
            self.sim_thread = SimulationThread(self.simulation_engine, config)
            self.sim_thread.progress_update.connect(self.update_progress)
            self.sim_thread.simulation_complete.connect(self.simulation_completed)
            self.sim_thread.error_occurred.connect(self.simulation_error)

            # Disable run button
            self.statusBar().showMessage("Simulation running...")

            # Start simulation
            self.sim_thread.start()

        except Exception as e:
            logger.error(f"Error setting up simulation: {str(e)}", exc_info=True)
            self.debug_console.append_message(f"Error: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to start simulation: {str(e)}")



    def update_progress(self, step):
        """Update progress bar and status during simulation"""
        self.statusBar().showMessage(f"Simulation step {step} in progress...")



    def simulation_error(self, error_msg):
        """Handle simulation errors"""
        self.debug_console.append_message(f"Simulation failed: {error_msg}", level='ERROR')
        self.statusBar().showMessage("Simulation failed")

        QMessageBox.critical(self, "Simulation Error", f"Simulation failed: {error_msg}")

    def plot_results(self, results):
        """Plot simulation results on the canvas"""
        try:
            # Clear the canvas
            self.canvas.axes.clear()

            # Check if we have valid results
            if results and 'plate_data' in results and 'time_points' in results:
                plate_data = results['plate_data']
                time_points = results['time_points']
                metadata = results.get('metadata', [])

                # Get number of wells
                num_wells = len(plate_data)

                # Define colors for different cell lines
                cell_line_colors = {
                    'Neurotypical': '#4DAF4A',  # Green
                    'ASD': '#984EA3',           # Purple
                    'FXS': '#FF7F00',           # Orange
                    'Default': '#999999'        # Gray
                }

                # Plot each well's trace with appropriate color
                for i in range(min(num_wells, 96)):  # Limit to 96 wells max for visibility
                    if i < len(metadata) and metadata[i].get('valid', True):
                        cell_line = metadata[i].get('cell_line', 'Default')
                        color = cell_line_colors.get(cell_line, cell_line_colors['Default'])

                        # Plot with semi-transparency to avoid overcrowding
                        self.canvas.axes.plot(time_points, plate_data[i], color=color, alpha=0.3, linewidth=0.8)

                # Add legend for cell lines
                for cell_line, color in cell_line_colors.items():
                    if cell_line != 'Default':  # Skip default in legend
                        self.canvas.axes.plot([], [], color=color, label=cell_line, linewidth=2)

                # Mark agonist addition time if available
                if 'params' in results and 'agonist_addition_time' in results['params']:
                    agonist_time = results['params']['agonist_addition_time']
                    self.canvas.axes.axvline(x=agonist_time, color='black', linestyle='--',
                                          label=f'Agonist ({agonist_time}s)')

                # Set labels and title
                agonist = results['params'].get('default_agonist', 'Unknown')
                concentration = results['params'].get('concentration', 0)
                concentration_unit = results['params'].get('concentration_unit', 'µM')

                self.canvas.axes.set_title(f'FLIPR Calcium Response: {agonist} ({concentration} {concentration_unit})')
                self.canvas.axes.set_xlabel('Time (s)')
                self.canvas.axes.set_ylabel('Fluorescence (A.U.)')
                self.canvas.axes.legend(loc='upper right')

            else:
                # If no valid results, show placeholder
                self.canvas.axes.text(0.5, 0.5, 'No valid simulation data available',
                                   ha='center', va='center', transform=self.canvas.axes.transAxes)
                self.canvas.axes.set_title('Simulation Results')

            # Refresh the canvas
            self.canvas.draw()

            # Log success
            self.debug_console.append_message("Plotted simulation results successfully")

        except Exception as e:
            # Log error
            error_msg = f"Error plotting results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.debug_console.append_message(error_msg, level='ERROR')

            # Show error in plot
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, f'Error plotting results:\n{str(e)}',
                               ha='center', va='center', transform=self.canvas.axes.transAxes)
            self.canvas.axes.set_title('Plot Error')
            self.canvas.draw()

    def on_well_selected(self, well_id):
        """Handle well selection from the grid"""
        # Uncheck all other buttons
        for wid, btn in self.well_buttons.items():
            if wid != well_id:
                btn.setChecked(False)

        # Make sure the selected button is checked
        self.well_buttons[well_id].setChecked(True)

        # Update the info label
        well_info = "Unknown"
        if hasattr(self, 'last_results') and 'metadata' in self.last_results:
            for meta in self.last_results['metadata']:
                if meta.get('well_id') == well_id:
                    cell_line = meta.get('cell_line', 'Unknown')
                    agonist = meta.get('agonist', 'Unknown')
                    conc = meta.get('concentration', 0)
                    well_info = f"{well_id}: {cell_line} with {agonist} ({conc} µM)"
                    break

        self.selected_well_label.setText(f"Selected: {well_info}")

        # Switch to Single Trace mode if not already there
        if self.plot_type_combo.currentText() != "Single Trace":
            self.plot_type_combo.setCurrentText("Single Trace")
        else:
            # If already in Single Trace mode, just update the plot
            self.show_individual_trace(well_id)

    # Add this new method to the MainWindow class
    def show_individual_trace(self, well_id=None):
        """Display a single trace for the selected well"""
        # If no well_id provided, find which one is selected in the grid
        if well_id is None:
            well_id = None
            for wid, btn in self.well_buttons.items():
                if btn.isChecked():
                    well_id = wid
                    break

        # If we have a well_id, plot it
        if well_id and hasattr(self, 'last_results'):
            self.plot_single_trace(self.last_results, well_id)

    # Add this new plotting function to the MainWindow class
    def plot_single_trace(self, results, well_id):
        """Plot a single trace for a specific well"""
        try:
            # Clear the canvas
            self.canvas.axes.clear()

            if not results or 'plate_data' not in results or 'metadata' not in results:
                self.canvas.axes.text(0.5, 0.5, 'No valid data available',
                                  ha='center', va='center', transform=self.canvas.axes.transAxes)
                self.canvas.draw()
                return

            # Find the well in metadata
            well_idx = None
            well_meta = None

            for i, meta in enumerate(results['metadata']):
                if meta.get('well_id', '') == well_id:
                    well_idx = i
                    well_meta = meta
                    break

            if well_idx is None or well_idx >= len(results['plate_data']):
                self.canvas.axes.text(0.5, 0.5, f'Well {well_id} not found in data',
                                  ha='center', va='center', transform=self.canvas.axes.transAxes)
                self.canvas.draw()
                return

            # Get data for this well
            time_points = results['time_points']
            well_data = results['plate_data'][well_idx]

            # Get well metadata
            cell_line = well_meta.get('cell_line', 'Unknown')
            agonist = well_meta.get('agonist', 'Unknown')
            concentration = well_meta.get('concentration', 0)

            # Determine line color based on cell line
            cell_line_colors = {
                'Neurotypical': '#4DAF4A',  # Green
                'ASD': '#984EA3',           # Purple
                'FXS': '#FF7F00',           # Orange
                'Unknown': '#999999'        # Gray
            }
            color = cell_line_colors.get(cell_line, cell_line_colors['Unknown'])

            # Plot the trace
            self.canvas.axes.plot(time_points, well_data, color=color, linewidth=2.5)

            # Add annotations
            if 'params' in results and 'agonist_addition_time' in results['params']:
                agonist_time = results['params']['agonist_addition_time']
                agonist_time_idx = int(agonist_time / results['params']['time_interval'])

                # Mark agonist addition
                self.canvas.axes.axvline(x=agonist_time, color='red', linestyle='--',
                                     label=f'Agonist Addition ({agonist_time}s)')

                # Calculate and mark baseline
                if agonist_time_idx > 0:
                    baseline = np.mean(well_data[:agonist_time_idx])
                    self.canvas.axes.axhline(y=baseline, color='green', linestyle=':',
                                         label=f'Baseline ({baseline:.1f})')

                    # Calculate and mark peak
                    if agonist_time_idx < len(well_data):
                        post_addition = well_data[agonist_time_idx:]
                        peak_local_idx = np.argmax(post_addition)
                        peak_idx = agonist_time_idx + peak_local_idx
                        peak_time = time_points[peak_idx]
                        peak_value = well_data[peak_idx]

                        # Mark peak point
                        self.canvas.axes.plot(peak_time, peak_value, 'ro', markersize=8,
                                          label=f'Peak ({peak_value:.1f})')

                        # Calculate peak response
                        peak_response = peak_value - baseline

                        # Add annotation
                        self.canvas.axes.annotate(f'Peak Response: {peak_response:.1f}',
                                              xy=(peak_time, peak_value),
                                              xytext=(peak_time + 10, peak_value + 50),
                                              arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

            # Set title and labels
            self.canvas.axes.set_title(f'Well {well_id}: {cell_line} with {agonist} ({concentration} µM)')
            self.canvas.axes.set_xlabel('Time (s)')
            self.canvas.axes.set_ylabel('Fluorescence (A.U.)')
            self.canvas.axes.legend()

            # Refresh canvas
            self.canvas.draw()

            # Log
            self.debug_console.append_message(f"Displaying trace for well {well_id}")

        except Exception as e:
            self.debug_console.append_message(f"Error plotting single trace: {str(e)}", level='ERROR')
            logger.error(f"Error plotting single trace: {str(e)}", exc_info=True)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
