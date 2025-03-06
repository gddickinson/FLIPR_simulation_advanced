import sys
import os
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                            QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                            QFileDialog, QMessageBox, QTextEdit, QGroupBox,
                            QFormLayout, QLineEdit, QSplitter, QFrame, QSizePolicy,
                            QTableWidgetItem, QTableWidget, QHeaderView, QGridLayout,
                            QRadioButton, QDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QTextCursor, QColor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import json
import re

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

        # First create the debug console so it's available to all tabs
        self.debug_console = DebugConsole()

        # Create tab widget
        self.tabs = QTabWidget()

        # Add tabs
        self.tabs.addTab(self.create_simulation_tab(), "Simulation")
        self.tabs.addTab(self.create_plate_layout_tab(), "Plate Layout")
        self.tabs.addTab(self.create_error_simulation_tab(), "Error Simulation")
        #self.tabs.addTab(self.create_batch_processing_tab(), "Batch Processing")
        self.tabs.addTab(self.create_debug_tab(), "Debug Console")
        self.tabs.addTab(self.create_settings_tab(), "Settings")

        # Add status bar
        self.statusBar().showMessage("Ready")

        # Assemble main layout
        main_layout.addWidget(self.tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Add a test message to the console
        self.debug_console.append_message("FLIPR Simulator initialized")

        # Log application start
        logger.info("FLIPR Simulator application started")


    def create_debug_tab(self):
        """Create the debug console tab"""
        tab = QWidget()
        layout = QVBoxLayout()

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
        self.read_noise_spin.setRange(0, 1000)
        self.read_noise_spin.setValue(20)
        noise_form.addRow("Read Noise:", self.read_noise_spin)

        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(0, 2000)
        self.background_spin.setValue(100)
        noise_form.addRow("Background:", self.background_spin)

        self.photobleaching_spin = QDoubleSpinBox()
        self.photobleaching_spin.setRange(0, 1.0)
        self.photobleaching_spin.setValue(0.0005)
        self.photobleaching_spin.setDecimals(6)
        self.photobleaching_spin.setSingleStep(0.0005)
        noise_form.addRow("Photobleaching Rate:", self.photobleaching_spin)

        # Add Reset to Defaults button
        self.reset_noise_btn = QPushButton("Reset to Defaults")
        self.reset_noise_btn.clicked.connect(self.reset_noise_parameters)
        noise_form.addRow("", self.reset_noise_btn)

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

        #plot control buttons
        plot_control_layout = QHBoxLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["All Traces", "By Cell Line", "By Agonist", "Heatmap", "Single Trace"])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot_type)

        self.export_plot_btn = QPushButton("Export Plot")
        self.export_plot_btn.clicked.connect(self.export_plot)

        self.export_data_btn = QPushButton("Export Data")
        self.export_data_btn.clicked.connect(self.export_simulation_data)
        self.export_data_btn.setEnabled(False)  # Will be enabled after simulation runs

        plot_control_layout.addWidget(QLabel("Plot Type:"))
        plot_control_layout.addWidget(self.plot_type_combo)
        plot_control_layout.addWidget(self.export_plot_btn)
        plot_control_layout.addWidget(self.export_data_btn)
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

            # Get active errors from error simulation tab
            if hasattr(self, 'get_active_errors'):
                active_errors = self.get_active_errors()
                if active_errors:
                    config['active_errors'] = active_errors

                    # Log the active errors
                    error_names = list(active_errors.keys())
                    self.debug_console.append_message(f"Including {len(error_names)} active errors: {', '.join(error_names)}")

            # Log configuration
            self.debug_console.append_message(f"Configuration: {config['num_timepoints']} timepoints, "
                                             f"{config['time_interval']}s interval, "
                                             f"agonist at {config['agonist_addition_time']}s")

            return config

        except Exception as e:
            self.debug_console.append_message(f"Error getting simulation config: {str(e)}", level='ERROR')
            logger.error(f"Error getting simulation config: {str(e)}", exc_info=True)
            return {}  # Return empty config on error


    def update_plot_type(self, plot_type):
        """Update the plot based on selected type"""
        if hasattr(self, 'last_results'):
            self.debug_console.append_message(f"Updating plot type to: {plot_type}")

            # Ensure we completely reset the figure and axes before creating a new plot
            self.canvas.fig.clear()
            self.canvas.axes = self.canvas.fig.add_subplot(111)

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

            # Ensure the canvas gets updated
            self.canvas.draw()

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

# Update the plot_heatmap method to properly clear colorbars

    def plot_heatmap(self, results):
        """Plot results as a plate heatmap"""
        try:
            # Clear the entire figure first to remove old colorbars
            self.canvas.fig.clear()

            # Create a new axis
            ax = self.canvas.fig.add_subplot(111)

            if not results or 'plate_data' not in results or 'params' not in results:
                ax.text(0.5, 0.5, 'No valid data for heatmap plot',
                       ha='center', va='center', transform=ax.transAxes)
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
            im = ax.imshow(peak_data, cmap='viridis')
            cbar = self.canvas.fig.colorbar(im, ax=ax, label='Peak Response (F-F0)')

            # Add well labels
            for i in range(rows):
                for j in range(cols):
                    ax.text(j, i, f"{chr(65+i)}{j+1}", ha='center', va='center',
                              color='white', fontsize=8)

            # Set title and labels
            ax.set_title('Peak Response Heatmap')

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

        # Enable export button
        if hasattr(self, 'export_data_btn'):
            self.export_data_btn.setEnabled(True)

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

        # Auto-save if enabled
        if hasattr(self, 'auto_save') and self.auto_save.isChecked():
            self.export_simulation_data()

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
        """Create the error simulation tab with interactive controls"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Introduction text
        intro_label = QLabel(
            "Simulate various error conditions that can occur in FLIPR experiments. "
            "Enable specific error types and adjust their probability and intensity. "
            "These errors will be applied when you run a simulation."
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        # Create a horizontal splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - error controls with tabs for standard and custom errors
        error_panel = QWidget()
        error_layout = QVBoxLayout()

        # Create a tab widget for standard vs custom errors
        error_tabs = QTabWidget()

        # Standard errors tab
        standard_tab = QWidget()
        standard_layout = QVBoxLayout()

        # Create error type groups
        # Cell-based errors
        cell_error_group = QGroupBox("Cell-Based Errors")
        cell_error_layout = QVBoxLayout()

        self.cell_variability_check = QCheckBox("Cell Variability")
        self.cell_variability_check.setToolTip("Introduces variability in cell responses")

        self.dye_loading_check = QCheckBox("Dye Loading Issues")
        self.dye_loading_check.setToolTip("Simulates problems with calcium dye loading")

        self.cell_health_check = QCheckBox("Cell Health Problems")
        self.cell_health_check.setToolTip("Simulates unhealthy cells with altered calcium response")

        self.cell_density_check = QCheckBox("Variable Cell Density")
        self.cell_density_check.setToolTip("Simulates uneven cell distribution across wells")

        cell_error_layout.addWidget(self.cell_variability_check)
        cell_error_layout.addWidget(self.dye_loading_check)
        cell_error_layout.addWidget(self.cell_health_check)
        cell_error_layout.addWidget(self.cell_density_check)

        cell_error_group.setLayout(cell_error_layout)
        standard_layout.addWidget(cell_error_group)

        # Reagent errors
        reagent_error_group = QGroupBox("Reagent-Based Errors")
        reagent_error_layout = QVBoxLayout()

        self.reagent_stability_check = QCheckBox("Reagent Stability Issues")
        self.reagent_stability_check.setToolTip("Simulates degraded reagents with reduced potency")

        self.reagent_concentration_check = QCheckBox("Incorrect Concentrations")
        self.reagent_concentration_check.setToolTip("Simulates pipetting errors causing concentration variations")

        self.reagent_contamination_check = QCheckBox("Reagent Contamination")
        self.reagent_contamination_check.setToolTip("Simulates contaminated reagents causing unexpected responses")

        reagent_error_layout.addWidget(self.reagent_stability_check)
        reagent_error_layout.addWidget(self.reagent_concentration_check)
        reagent_error_layout.addWidget(self.reagent_contamination_check)

        reagent_error_group.setLayout(reagent_error_layout)
        standard_layout.addWidget(reagent_error_group)

        # Equipment errors
        equipment_error_group = QGroupBox("Equipment-Based Errors")
        equipment_error_layout = QVBoxLayout()

        self.camera_error_check = QCheckBox("Camera Errors")
        self.camera_error_check.setToolTip("Simulates camera artifacts and errors")

        self.liquid_handler_check = QCheckBox("Liquid Handler Issues")
        self.liquid_handler_check.setToolTip("Simulates inaccurate dispensing of reagents")

        self.timing_error_check = QCheckBox("Timing Inconsistencies")
        self.timing_error_check.setToolTip("Simulates timing issues with data collection")

        self.focus_error_check = QCheckBox("Focus Problems")
        self.focus_error_check.setToolTip("Simulates focus issues affecting signal quality")

        equipment_error_layout.addWidget(self.camera_error_check)
        equipment_error_layout.addWidget(self.liquid_handler_check)
        equipment_error_layout.addWidget(self.timing_error_check)
        equipment_error_layout.addWidget(self.focus_error_check)

        equipment_error_group.setLayout(equipment_error_layout)
        standard_layout.addWidget(equipment_error_group)

        # Systematic errors
        systematic_error_group = QGroupBox("Systematic Errors")
        systematic_error_layout = QVBoxLayout()

        self.edge_effect_check = QCheckBox("Plate Edge Effects")
        self.edge_effect_check.setToolTip("Simulates edge effects common in microplates")

        self.temperature_check = QCheckBox("Temperature Gradients")
        self.temperature_check.setToolTip("Simulates temperature variations across the plate")

        self.evaporation_check = QCheckBox("Evaporation")
        self.evaporation_check.setToolTip("Simulates evaporation effects over time")

        self.well_crosstalk_check = QCheckBox("Well-to-Well Crosstalk")
        self.well_crosstalk_check.setToolTip("Simulates optical crosstalk between adjacent wells")

        systematic_error_layout.addWidget(self.edge_effect_check)
        systematic_error_layout.addWidget(self.temperature_check)
        systematic_error_layout.addWidget(self.evaporation_check)
        systematic_error_layout.addWidget(self.well_crosstalk_check)

        systematic_error_group.setLayout(systematic_error_layout)
        standard_layout.addWidget(systematic_error_group)

        standard_tab.setLayout(standard_layout)
        error_tabs.addTab(standard_tab, "Standard Errors")

        # Custom errors tab
        custom_tab = QWidget()
        custom_layout = QVBoxLayout()

        # Custom error enable checkbox
        self.custom_error_check = QCheckBox("Enable Custom Error")
        self.custom_error_check.setToolTip("Enable a highly customizable error type")
        custom_layout.addWidget(self.custom_error_check)

        # Custom error type selector
        custom_type_layout = QFormLayout()
        self.custom_error_type = QComboBox()
        self.custom_error_type.addItems([
            "Random Spikes",
            "Signal Dropouts",
            "Baseline Drift",
            "Oscillating Baseline",
            "Signal Cutout",
            "Incomplete Decay",
            "Extra Noise",
            "Overlapping Oscillation",
            "Sudden Jump",
            "Exponential Drift",
            "Delayed Response"
        ])
        self.custom_error_type.currentTextChanged.connect(self.update_custom_error_params)
        custom_type_layout.addRow("Error Type:", self.custom_error_type)
        custom_layout.addLayout(custom_type_layout)

        # Custom error parameters - will be dynamically updated
        self.custom_params_group = QGroupBox("Error Parameters")
        self.custom_params_layout = QFormLayout()
        self.custom_params_group.setLayout(self.custom_params_layout)
        custom_layout.addWidget(self.custom_params_group)

        # Well selection for custom error
        wells_group = QGroupBox("Apply To Wells")
        wells_layout = QVBoxLayout()

        self.custom_all_wells_radio = QRadioButton("All Wells")
        self.custom_all_wells_radio.setChecked(True)

        self.custom_specific_wells_radio = QRadioButton("Specific Wells:")

        self.custom_wells_edit = QLineEdit("A1,B2,C3")
        self.custom_wells_edit.setEnabled(False)
        self.custom_wells_edit.setToolTip("Comma-separated list of wells (e.g., 'A1,B2,C3')")

        self.custom_specific_wells_radio.toggled.connect(
            lambda checked: self.custom_wells_edit.setEnabled(checked))

        wells_layout.addWidget(self.custom_all_wells_radio)
        wells_layout.addWidget(self.custom_specific_wells_radio)
        wells_layout.addWidget(self.custom_wells_edit)

        wells_group.setLayout(wells_layout)
        custom_layout.addWidget(wells_group)


        # Add the new checkbox for using global error settings
        self.use_global_settings_check = QCheckBox("Use Global Error Settings")
        self.use_global_settings_check.setToolTip("Apply global error probability and intensity settings to this custom error")
        self.use_global_settings_check.setChecked(False)  # Default to unchecked
        custom_layout.addWidget(self.use_global_settings_check)


        # # Preview button
        # self.preview_custom_error_btn = QPushButton("Preview Custom Error")
        # self.preview_custom_error_btn.clicked.connect(self.preview_custom_error)
        # custom_layout.addWidget(self.preview_custom_error_btn)

        custom_layout.addStretch()
        custom_tab.setLayout(custom_layout)
        error_tabs.addTab(custom_tab, "Custom Errors")

        error_layout.addWidget(error_tabs)

        # Global error settings
        global_settings_group = QGroupBox("Global Error Settings")
        global_settings_layout = QFormLayout()

        self.error_probability_spin = QDoubleSpinBox()
        self.error_probability_spin.setRange(0, 1)
        self.error_probability_spin.setValue(0.5)
        self.error_probability_spin.setSingleStep(0.05)
        self.error_probability_spin.setDecimals(2)
        global_settings_layout.addRow("Error Probability:", self.error_probability_spin)

        self.error_intensity_spin = QDoubleSpinBox()
        self.error_intensity_spin.setRange(0, 1)
        self.error_intensity_spin.setValue(0.5)
        self.error_intensity_spin.setSingleStep(0.05)
        self.error_intensity_spin.setDecimals(2)
        global_settings_layout.addRow("Error Intensity:", self.error_intensity_spin)

        global_settings_group.setLayout(global_settings_layout)
        error_layout.addWidget(global_settings_group)

        # Preset scenarios dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset Error Scenarios:"))

        self.error_preset_combo = QComboBox()
        self.error_preset_combo.addItems([
            "Custom Settings",
            "Dye Loading Issues",
            "Cell Health Problems",
            "Liquid Handler Failure",
            "Edge Effects",
            "Camera Failure",
            "Reagent Degradation",
            "Combined Failures"
        ])
        self.error_preset_combo.currentTextChanged.connect(self.apply_error_preset)
        preset_layout.addWidget(self.error_preset_combo, 1)

        error_layout.addLayout(preset_layout)

        # Control buttons
        button_layout = QHBoxLayout()

        self.apply_errors_btn = QPushButton("Apply Error Settings")
        self.apply_errors_btn.clicked.connect(self.apply_error_settings)

        self.run_comparison_btn = QPushButton("Run Error Comparison")
        self.run_comparison_btn.clicked.connect(self.run_error_comparison)
        self.run_comparison_btn.setToolTip("Run a comparison between normal and error-affected simulations")

        self.clear_errors_btn = QPushButton("Clear All Errors")
        self.clear_errors_btn.clicked.connect(self.clear_error_settings)

        button_layout.addWidget(self.apply_errors_btn)
        button_layout.addWidget(self.run_comparison_btn)
        button_layout.addWidget(self.clear_errors_btn)
        button_layout.addStretch()

        error_layout.addLayout(button_layout)
        error_layout.addStretch()

        error_panel.setLayout(error_layout)

        # Right panel - visualization
        viz_panel = QWidget()
        viz_layout = QVBoxLayout()

        # Error visualization section
        self.error_canvas = MatplotlibCanvas(viz_panel, width=6, height=5, dpi=100)
        viz_layout.addWidget(self.error_canvas)

        # Description of currently selected errors
        self.error_description = QTextEdit()
        self.error_description.setReadOnly(True)
        self.error_description.setMinimumHeight(150)
        self.error_description.setText(
            "No errors selected. Use the checkboxes to enable specific error types or select a preset scenario."
        )
        viz_layout.addWidget(QLabel("Active Error Description:"))
        viz_layout.addWidget(self.error_description)

        viz_panel.setLayout(viz_layout)

        # Add panels to splitter
        splitter.addWidget(error_panel)
        splitter.addWidget(viz_panel)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)
        tab.setLayout(layout)

        # Initialize with all errors disabled
        self.clear_error_settings()

        # Initialize custom error parameters
        self.update_custom_error_params(self.custom_error_type.currentText())

        return tab

    def update_custom_error_params(self, error_type):
        """Update the custom error parameters based on the selected error type"""
        # Clear the current parameters by directly getting and removing all widgets
        for i in reversed(range(self.custom_params_layout.count())):
            widget = self.custom_params_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
                self.custom_params_layout.removeWidget(widget)

        # Add new parameters based on error type
        if error_type == "Random Spikes":
            self.num_spikes_spin = QSpinBox()
            self.num_spikes_spin.setRange(1, 10)
            self.num_spikes_spin.setValue(3)
            self.custom_params_layout.addRow("Number of Spikes:", self.num_spikes_spin)

            self.spike_amplitude_spin = QDoubleSpinBox()
            self.spike_amplitude_spin.setRange(100, 5000)
            self.spike_amplitude_spin.setValue(1000)
            self.spike_amplitude_spin.setSingleStep(100)
            self.custom_params_layout.addRow("Spike Amplitude:", self.spike_amplitude_spin)

            self.spike_width_spin = QSpinBox()
            self.spike_width_spin.setRange(1, 10)
            self.spike_width_spin.setValue(3)
            self.custom_params_layout.addRow("Spike Width:", self.spike_width_spin)

        elif error_type == "Signal Dropouts":
            self.num_dropouts_spin = QSpinBox()
            self.num_dropouts_spin.setRange(1, 5)
            self.num_dropouts_spin.setValue(2)
            self.custom_params_layout.addRow("Number of Dropouts:", self.num_dropouts_spin)

            self.dropout_length_spin = QSpinBox()
            self.dropout_length_spin.setRange(5, 50)
            self.dropout_length_spin.setValue(10)
            self.custom_params_layout.addRow("Dropout Length:", self.dropout_length_spin)

            self.dropout_factor_spin = QDoubleSpinBox()
            self.dropout_factor_spin.setRange(0, 0.5)
            self.dropout_factor_spin.setValue(0.1)
            self.dropout_factor_spin.setSingleStep(0.05)
            self.custom_params_layout.addRow("Remaining Signal (%):", self.dropout_factor_spin)

        elif error_type == "Baseline Drift":
            self.drift_direction_combo = QComboBox()
            self.drift_direction_combo.addItems(["Rising", "Falling"])
            self.custom_params_layout.addRow("Drift Direction:", self.drift_direction_combo)

            self.drift_magnitude_spin = QDoubleSpinBox()
            self.drift_magnitude_spin.setRange(100, 2000)
            self.drift_magnitude_spin.setValue(500)
            self.drift_magnitude_spin.setSingleStep(100)
            self.custom_params_layout.addRow("Drift Magnitude:", self.drift_magnitude_spin)

        elif error_type == "Oscillating Baseline":
            self.oscillation_freq_spin = QDoubleSpinBox()
            self.oscillation_freq_spin.setRange(0.01, 0.2)
            self.oscillation_freq_spin.setValue(0.05)
            self.oscillation_freq_spin.setSingleStep(0.01)
            self.oscillation_freq_spin.setDecimals(3)
            self.custom_params_layout.addRow("Oscillation Frequency:", self.oscillation_freq_spin)

            self.oscillation_amp_spin = QDoubleSpinBox()
            self.oscillation_amp_spin.setRange(50, 1000)
            self.oscillation_amp_spin.setValue(300)
            self.oscillation_amp_spin.setSingleStep(50)
            self.custom_params_layout.addRow("Oscillation Amplitude:", self.oscillation_amp_spin)

        elif error_type == "Signal Cutout":
            self.cutout_start_spin = QDoubleSpinBox()
            self.cutout_start_spin.setRange(0.1, 0.9)
            self.cutout_start_spin.setValue(0.3)
            self.cutout_start_spin.setSingleStep(0.05)
            self.cutout_start_spin.setDecimals(2)
            self.custom_params_layout.addRow("Start Position (% of trace):", self.cutout_start_spin)

            self.cutout_duration_spin = QDoubleSpinBox()
            self.cutout_duration_spin.setRange(0.05, 0.5)
            self.cutout_duration_spin.setValue(0.2)
            self.cutout_duration_spin.setSingleStep(0.05)
            self.cutout_duration_spin.setDecimals(2)
            self.custom_params_layout.addRow("Duration (% of trace):", self.cutout_duration_spin)

            self.replace_baseline_check = QCheckBox("Replace with baseline (otherwise zeros)")
            self.replace_baseline_check.setChecked(True)
            self.custom_params_layout.addRow("", self.replace_baseline_check)

        elif error_type == "Incomplete Decay":
            self.elevation_factor_spin = QDoubleSpinBox()
            self.elevation_factor_spin.setRange(0.1, 0.9)
            self.elevation_factor_spin.setValue(0.5)
            self.elevation_factor_spin.setSingleStep(0.05)
            self.elevation_factor_spin.setDecimals(2)
            self.custom_params_layout.addRow("Elevation Factor:", self.elevation_factor_spin)

        elif error_type == "Extra Noise":
            self.noise_std_spin = QDoubleSpinBox()
            self.noise_std_spin.setRange(20, 500)
            self.noise_std_spin.setValue(100)
            self.noise_std_spin.setSingleStep(10)
            self.custom_params_layout.addRow("Noise Standard Deviation:", self.noise_std_spin)

        elif error_type == "Overlapping Oscillation":
            self.overlapping_freq_spin = QDoubleSpinBox()
            self.overlapping_freq_spin.setRange(0.01, 0.5)
            self.overlapping_freq_spin.setValue(0.1)
            self.overlapping_freq_spin.setSingleStep(0.01)
            self.overlapping_freq_spin.setDecimals(3)
            self.custom_params_layout.addRow("Oscillation Frequency:", self.overlapping_freq_spin)

            self.overlapping_amp_spin = QDoubleSpinBox()
            self.overlapping_amp_spin.setRange(50, 1000)
            self.overlapping_amp_spin.setValue(200)
            self.overlapping_amp_spin.setSingleStep(50)
            self.custom_params_layout.addRow("Oscillation Amplitude:", self.overlapping_amp_spin)

            self.phase_shift_spin = QDoubleSpinBox()
            self.phase_shift_spin.setRange(0, 6.28)
            self.phase_shift_spin.setValue(0)
            self.phase_shift_spin.setSingleStep(0.5)
            self.phase_shift_spin.setDecimals(2)
            self.custom_params_layout.addRow("Phase Shift (radians):", self.phase_shift_spin)

        elif error_type == "Sudden Jump":
            self.jump_position_spin = QDoubleSpinBox()
            self.jump_position_spin.setRange(0.1, 0.9)
            self.jump_position_spin.setValue(0.7)
            self.jump_position_spin.setSingleStep(0.05)
            self.jump_position_spin.setDecimals(2)
            self.custom_params_layout.addRow("Jump Position (% of trace):", self.jump_position_spin)

            self.jump_magnitude_spin = QDoubleSpinBox()
            self.jump_magnitude_spin.setRange(-2000, 2000)
            self.jump_magnitude_spin.setValue(500)
            self.jump_magnitude_spin.setSingleStep(100)
            self.custom_params_layout.addRow("Jump Magnitude:", self.jump_magnitude_spin)

        elif error_type == "Exponential Drift":
            self.exp_direction_combo = QComboBox()
            self.exp_direction_combo.addItems(["Upward", "Downward"])
            self.custom_params_layout.addRow("Drift Direction:", self.exp_direction_combo)

            self.exp_magnitude_spin = QDoubleSpinBox()
            self.exp_magnitude_spin.setRange(100, 3000)
            self.exp_magnitude_spin.setValue(1000)
            self.exp_magnitude_spin.setSingleStep(100)
            self.custom_params_layout.addRow("Maximum Magnitude:", self.exp_magnitude_spin)

            self.exp_rate_spin = QDoubleSpinBox()
            self.exp_rate_spin.setRange(1, 10)
            self.exp_rate_spin.setValue(3)
            self.exp_rate_spin.setSingleStep(0.5)
            self.exp_rate_spin.setDecimals(1)
            self.custom_params_layout.addRow("Exponential Rate:", self.exp_rate_spin)

        elif error_type == "Delayed Response":
            self.delay_time_spin = QDoubleSpinBox()
            self.delay_time_spin.setRange(1, 20)
            self.delay_time_spin.setValue(5)
            self.delay_time_spin.setSingleStep(1)
            self.delay_time_spin.setDecimals(1)
            self.custom_params_layout.addRow("Delay Time (seconds):", self.delay_time_spin)

    def preview_custom_error(self):
        """Generate a preview of the custom error effect"""
        try:
            if not hasattr(self, 'simulation_engine'):
                QMessageBox.warning(self, "Warning", "Simulation engine not initialized.")
                return

            # Generate a sample calcium response
            time_points = np.arange(0, 180, 0.4)  # 451 timepoints at 0.4s interval

            # Create a sample params dictionary
            params = {
                'time_interval': 0.4,
                'agonist_addition_time': 10
            }

            # Create a sample normal calcium response
            baseline = 500
            peak = 3000
            response = np.ones_like(time_points) * baseline

            # Add a peak after agonist addition
            agonist_idx = int(10 / 0.4)  # 10s divided by 0.4s interval
            peak_idx = agonist_idx + 10  # Peak 4s after agonist

            # Create rising phase
            for i in range(agonist_idx, peak_idx + 1):
                if i < len(response):
                    progress = (i - agonist_idx) / (peak_idx - agonist_idx)
                    response[i] = baseline + progress * (peak - baseline)

            # Create decay phase
            decay_rate = 0.05
            for i in range(peak_idx + 1, len(response)):
                time_since_peak = (i - peak_idx) * 0.4
                response[i] = baseline + (peak - baseline) * np.exp(-decay_rate * time_since_peak)

            # Get custom error parameters based on selected type
            error_type = self.custom_error_type.currentText()
            error_params = self.get_custom_error_params()

            # Check if we should use global settings
            use_global = False
            if hasattr(self, 'use_global_settings_check'):
                use_global = self.use_global_settings_check.isChecked()

            # Get global settings if needed
            if use_global:
                intensity = self.error_intensity_spin.value()
                # Update error params with global intensity if needed
                error_params = self.get_custom_error_params()
                if 'intensity' in error_params:
                    error_params['intensity'] = intensity


            # Apply the custom error
            settings = {
                'custom_type': error_type.lower().replace(' ', '_'),
                'custom_params': error_params,
                'use_global_settings': use_global
            }

            altered_response = self.simulation_engine._apply_custom_error(
                response, 0, 0, 0, params, settings
            )

            # Clear the canvas
            self.error_canvas.axes.clear()

            # Plot original and altered responses
            self.error_canvas.axes.plot(time_points, response, 'b-', label='Normal', linewidth=2)
            self.error_canvas.axes.plot(time_points, altered_response, 'r-', label='With Error', linewidth=2)

            # Mark agonist addition
            self.error_canvas.axes.axvline(x=10, color='k', linestyle='--', label='Agonist Addition')

            # Set labels and title
            self.error_canvas.axes.set_title(f'Custom Error Preview: {error_type}')
            self.error_canvas.axes.set_xlabel('Time (s)')
            self.error_canvas.axes.set_ylabel('Fluorescence (A.U.)')
            self.error_canvas.axes.legend()

            # Refresh canvas
            self.error_canvas.draw()

            # Update description
            self.error_description.setText(
                f"Custom Error Preview: {error_type}\n\n"
                f"This shows how the selected custom error would affect a typical calcium response.\n\n"
                f"Parameters:\n{self.format_error_params(error_params)}\n\n"
                f"Apply this error by enabling it in the simulation settings."
            )

        except Exception as e:
            self.debug_console.append_message(f"Error previewing custom error: {str(e)}", level='ERROR')
            logger.error(f"Error previewing custom error: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to preview custom error: {str(e)}")

    def get_custom_error_params(self):
        """Get the parameter values for the current custom error type"""
        error_type = self.custom_error_type.currentText()
        params = {}

        if error_type == "Random Spikes":
            params['num_spikes'] = self.num_spikes_spin.value()
            params['amplitude'] = self.spike_amplitude_spin.value()
            params['width'] = self.spike_width_spin.value()

        elif error_type == "Signal Dropouts":
            params['num_dropouts'] = self.num_dropouts_spin.value()
            params['length'] = self.dropout_length_spin.value()
            params['factor'] = self.dropout_factor_spin.value()

        elif error_type == "Baseline Drift":
            params['direction'] = self.drift_direction_combo.currentText().lower()
            params['magnitude'] = self.drift_magnitude_spin.value()

        elif error_type == "Oscillating Baseline":
            params['frequency'] = self.oscillation_freq_spin.value()
            params['amplitude'] = self.oscillation_amp_spin.value()

        elif error_type == "Signal Cutout":
            params['start_pct'] = self.cutout_start_spin.value()
            params['duration_pct'] = self.cutout_duration_spin.value()
            params['replace_with_baseline'] = self.replace_baseline_check.isChecked()

        elif error_type == "Incomplete Decay":
            params['elevation_factor'] = self.elevation_factor_spin.value()

        elif error_type == "Extra Noise":
            params['std'] = self.noise_std_spin.value()

        elif error_type == "Overlapping Oscillation":
            params['frequency'] = self.overlapping_freq_spin.value()
            params['amplitude'] = self.overlapping_amp_spin.value()
            params['phase_shift'] = self.phase_shift_spin.value()

        elif error_type == "Sudden Jump":
            params['position_pct'] = self.jump_position_spin.value()
            params['magnitude'] = self.jump_magnitude_spin.value()

        elif error_type == "Exponential Drift":
            params['direction'] = self.exp_direction_combo.currentText().lower()
            params['magnitude'] = self.exp_magnitude_spin.value()
            params['rate'] = self.exp_rate_spin.value()

        elif error_type == "Delayed Response":
            params['delay_seconds'] = self.delay_time_spin.value()

        return params

    def format_error_params(self, params):
        """Format error parameters for display"""
        result = ""
        for key, value in params.items():
            # Convert snake_case to Title Case with spaces
            display_key = ' '.join(word.capitalize() for word in key.split('_'))
            result += f"• {display_key}: {value}\n"
        return result

    def apply_error_preset(self, preset_name):
        """Apply a preset error scenario"""
        if preset_name == "Custom Settings":
            return  # Don't change current settings

        # First clear all errors
        self.clear_error_settings(update_ui=False)

        # Set the global error settings
        self.error_probability_spin.setValue(0.5)
        self.error_intensity_spin.setValue(0.6)

        # Apply specific preset
        if preset_name == "Dye Loading Issues":
            self.dye_loading_check.setChecked(True)
            self.error_description.setText(
                "Dye Loading Issues: Simulates problems with calcium dye loading in cells.\n\n"
                "This results in reduced signal amplitude and altered baseline fluorescence. "
                "Common causes include incomplete AM ester hydrolysis, dye compartmentalization, "
                "or uneven loading across the cell population."
            )

        elif preset_name == "Cell Health Problems":
            self.cell_health_check.setChecked(True)
            self.error_description.setText(
                "Cell Health Problems: Simulates unhealthy cells with altered calcium responses.\n\n"
                "Unhealthy cells typically show higher baseline calcium, reduced peak responses, "
                "and slower decay rates. This can be caused by cell stress, contamination, or "
                "inappropriate culture conditions."
            )

        elif preset_name == "Liquid Handler Failure":
            self.liquid_handler_check.setChecked(True)
            self.error_probability_spin.setValue(0.7)
            self.error_description.setText(
                "Liquid Handler Failure: Simulates problems with agonist addition.\n\n"
                "This includes inaccurate dispensing, timing errors, missed wells, or double additions. "
                "Results in missing, delayed, or abnormal calcium responses that don't reflect the "
                "true biology of the cells."
            )

        elif preset_name == "Edge Effects":
            self.edge_effect_check.setChecked(True)
            self.error_probability_spin.setValue(1.0)  # Always affects edge wells
            self.error_description.setText(
                "Edge Effects: Simulates microplate edge effects.\n\n"
                "Wells at the plate edges often show different behavior due to thermal gradients, "
                "evaporation, or optical effects. This typically results in higher variability and "
                "systematic differences between edge and interior wells."
            )

        elif preset_name == "Camera Failure":
            self.camera_error_check.setChecked(True)
            self.error_description.setText(
                "Camera Failure: Simulates camera artifacts and errors.\n\n"
                "These include dead pixels, saturation, noise spikes, and signal drops. "
                "Camera issues can result in artificial spikes, plateaus, or data gaps "
                "that aren't related to actual calcium signaling."
            )

        elif preset_name == "Reagent Degradation":
            self.reagent_stability_check.setChecked(True)
            self.error_probability_spin.setValue(0.8)
            self.error_description.setText(
                "Reagent Degradation: Simulates degraded reagents with reduced potency.\n\n"
                "Old or improperly stored reagents can lose activity, resulting in weaker responses. "
                "This typically causes reduced peak height while maintaining normal kinetics."
            )

        elif preset_name == "Combined Failures":
            self.edge_effect_check.setChecked(True)
            self.liquid_handler_check.setChecked(True)
            self.cell_variability_check.setChecked(True)
            self.error_probability_spin.setValue(0.4)
            self.error_description.setText(
                "Combined Failures: Multiple errors occurring simultaneously.\n\n"
                "Real experiments often suffer from multiple error types at once. This preset "
                "combines edge effects, liquid handler issues, and cell variability to create "
                "a challenging but realistic error scenario."
            )

        # Update the UI to show active errors
        self.update_error_display()

    def apply_error_settings(self):
        """Apply current error settings and update the description"""
        error_text = "Active Errors:\n"
        active_errors = self.get_active_errors()

        if not active_errors:
            error_text = "No errors selected. The simulation will run without introducing artificial errors."
        else:
            for error_type in active_errors:
                if error_type == 'cell_variability':
                    error_text += "• Cell Variability: Increased variability in cell responses\n"
                elif error_type == 'dye_loading':
                    error_text += "• Dye Loading Issues: Problems with calcium dye loading\n"
                elif error_type == 'cell_health':
                    error_text += "• Cell Health Problems: Unhealthy cells with altered responses\n"
                elif error_type == 'cell_density':
                    error_text += "• Variable Cell Density: Uneven cell distribution\n"
                elif error_type == 'reagent_stability':
                    error_text += "• Reagent Stability Issues: Degraded reagents\n"
                elif error_type == 'reagent_concentration':
                    error_text += "• Incorrect Concentrations: Pipetting errors\n"
                elif error_type == 'reagent_contamination':
                    error_text += "• Reagent Contamination: Contaminated reagents\n"
                elif error_type == 'camera_errors':
                    error_text += "• Camera Errors: Camera artifacts and errors\n"
                elif error_type == 'liquid_handler':
                    error_text += "• Liquid Handler Issues: Inaccurate dispensing\n"
                elif error_type == 'timing_errors':
                    error_text += "• Timing Inconsistencies: Timing issues\n"
                elif error_type == 'focus_problems':
                    error_text += "• Focus Problems: Focus issues affecting signal\n"
                elif error_type == 'edge_effects':
                    error_text += "• Edge Effects: Plate edge effects\n"
                elif error_type == 'temperature_gradient':
                    error_text += "• Temperature Gradients: Temperature variations\n"
                elif error_type == 'evaporation':
                    error_text += "• Evaporation: Evaporation effects\n"
                elif error_type == 'well_crosstalk':
                    error_text += "• Well Crosstalk: Optical crosstalk between wells\n"

            error_text += f"\nError Probability: {self.error_probability_spin.value()}\n"
            error_text += f"Error Intensity: {self.error_intensity_spin.value()}\n"
            error_text += "\nThese errors will be applied in the next simulation run."

        self.error_description.setText(error_text)
        self.update_error_display()

        # Set the error preset combo back to "Custom Settings"
        self.error_preset_combo.setCurrentText("Custom Settings")

        # Log the applied errors
        self.debug_console.append_message(f"Applied error settings with {len(active_errors)} active errors")

    # Update the clear_error_settings method to include custom errors
    def clear_error_settings(self, update_ui=True):
        """Clear all error settings"""
        # Uncheck all error checkboxes
        for attr_name in dir(self):
            if attr_name.endswith('_check') and isinstance(getattr(self, attr_name), QCheckBox):
                getattr(self, attr_name).setChecked(False)

        # Reset probability and intensity
        self.error_probability_spin.setValue(0.5)
        self.error_intensity_spin.setValue(0.5)

        # Reset custom error settings
        if hasattr(self, 'custom_all_wells_radio'):
            self.custom_all_wells_radio.setChecked(True)

        if hasattr(self, 'custom_wells_edit'):
            self.custom_wells_edit.setText("A1,B2,C3")

        if update_ui:
            # Update description
            self.error_description.setText(
                "No errors selected. The simulation will run without introducing artificial errors."
            )

            # Update the error display
            self.update_error_display()

            # Set the error preset combo to "Custom Settings"
            self.error_preset_combo.setCurrentText("Custom Settings")

            # Log
            self.debug_console.append_message("Cleared all error settings")

    # Update the get_active_errors method to include custom errors
    def get_active_errors(self):
        """Get a dictionary of active error types and their settings"""
        active_errors = {}

        # Map checkbox attributes to error model names
        error_map = {
            'cell_variability_check': 'cell_variability',
            'dye_loading_check': 'dye_loading',
            'cell_health_check': 'cell_health',
            'cell_density_check': 'cell_density',
            'reagent_stability_check': 'reagent_stability',
            'reagent_concentration_check': 'reagent_concentration',
            'reagent_contamination_check': 'reagent_contamination',
            'camera_error_check': 'camera_errors',
            'liquid_handler_check': 'liquid_handler',
            'timing_error_check': 'timing_errors',
            'focus_error_check': 'focus_problems',
            'edge_effect_check': 'edge_effects',
            'temperature_check': 'temperature_gradient',
            'evaporation_check': 'evaporation',
            'well_crosstalk_check': 'well_crosstalk'
        }

        # Get probability and intensity values
        probability = self.error_probability_spin.value()
        intensity = self.error_intensity_spin.value()

        # Check which errors are enabled
        for checkbox_name, error_name in error_map.items():
            if hasattr(self, checkbox_name) and getattr(self, checkbox_name).isChecked():
                active_errors[error_name] = {
                    'active': True,
                    'probability': probability,
                    'intensity': intensity
                }

                # Special case for temperature gradient
                if error_name == 'temperature_gradient':
                    active_errors[error_name]['pattern'] = 'left-to-right'

        # Add custom error if enabled
        if hasattr(self, 'custom_error_check') and self.custom_error_check.isChecked():
            error_type = self.custom_error_type.currentText()
            error_params = self.get_custom_error_params()

            # Get well selection
            wells = []
            if hasattr(self, 'custom_specific_wells_radio') and self.custom_specific_wells_radio.isChecked():
                well_text = self.custom_wells_edit.text()
                wells = [w.strip() for w in well_text.split(',') if w.strip()]

            # Check if we should use global settings
            use_global = False
            if hasattr(self, 'use_global_settings_check'):
                use_global = self.use_global_settings_check.isChecked()

            active_errors['custom_error'] = {
                'active': True,
                'probability': probability if use_global else 1.0,  # Use global probability if checked, otherwise always apply
                'intensity': intensity if use_global else self.error_intensity_spin.value(),  # Use global intensity if checked
                'custom_type': error_type.lower().replace(' ', '_'),
                'custom_params': error_params,
                'specific_wells': wells if wells else None,
                'use_global_settings': use_global  # Add this flag for reference
            }

        return active_errors

    def update_error_display(self):
        """Update the error visualization"""
        try:
            # Clear the entire figure first to remove old colorbars
            self.error_canvas.fig.clear()

            # Create a new axis
            ax = self.error_canvas.fig.add_subplot(111)

            active_errors = self.get_active_errors()

            if not active_errors:
                # Show a placeholder message
                ax.text(0.5, 0.5, 'No errors selected',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=14)
                ax.set_axis_off()
                self.error_canvas.draw()
                return

            # Create a plate visualization showing which wells would be affected
            plate = np.zeros((8, 12))

            # Apply error probability to simulate which wells would be affected
            for error_type, settings in active_errors.items():
                probability = settings['probability']

                if error_type == 'edge_effects':
                    # Edge effects primarily affect outer wells
                    for i in range(8):
                        for j in range(12):
                            if i == 0 or i == 7 or j == 0 or j == 11:  # Edge wells
                                if np.random.random() < probability:
                                    plate[i, j] += 1

                elif error_type == 'temperature_gradient':
                    # Temperature gradient affects wells based on position
                    pattern = settings.get('pattern', 'left-to-right')

                    if pattern == 'left-to-right':
                        # Gradient from left to right
                        for i in range(8):
                            for j in range(12):
                                if np.random.random() < probability * (j+1)/12:
                                    plate[i, j] += 0.5

                    elif pattern == 'center-to-edge':
                        # Gradient from center to edge
                        center_i, center_j = 3.5, 5.5
                        for i in range(8):
                            for j in range(12):
                                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                                max_dist = np.sqrt(center_i**2 + center_j**2)
                                if np.random.random() < probability * dist/max_dist:
                                    plate[i, j] += 0.5

                else:
                    # Other errors affect wells randomly based on probability
                    for i in range(8):
                        for j in range(12):
                            if np.random.random() < probability:
                                plate[i, j] += 0.75

            # Normalize plate values
            if np.max(plate) > 0:
                plate = plate / np.max(plate)

            # Create heatmap
            im = ax.imshow(plate, cmap='YlOrRd', interpolation='nearest')
            self.error_canvas.fig.colorbar(im, ax=ax, label='Error Probability')

            # Add well labels
            for i in range(8):
                for j in range(12):
                    ax.text(j, i, f"{chr(65+i)}{j+1}", ha='center', va='center',
                                       color='black', fontsize=8)

            # Set title and labels
            active_error_count = len(active_errors)
            ax.set_title(f'Error Distribution ({active_error_count} active error types)')

            # Refresh canvas
            self.error_canvas.draw()

        except Exception as e:
            self.debug_console.append_message(f"Error updating error display: {str(e)}", level='ERROR')

    # def create_batch_processing_tab(self):
    #     """Create the batch processing tab"""
    #     tab = QWidget()
    #     layout = QVBoxLayout()

    #     # TODO: Implement batch processing interface
    #     placeholder = QLabel("Batch Processing will be implemented here")
    #     placeholder.setAlignment(Qt.AlignCenter)

    #     layout.addWidget(placeholder)
    #     tab.setLayout(layout)

    #     return tab

    def create_settings_tab(self):
        """Create the settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Application settings form
        settings_form = QFormLayout()

        # Output directory setting
        self.output_dir_edit = QLineEdit(os.path.join(os.getcwd(), "simulation_results"))
        output_dir_btn = QPushButton("Browse...")
        output_dir_btn.clicked.connect(self.browse_output_directory)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(output_dir_btn)

        settings_form.addRow("Output Directory:", output_dir_layout)

        # Default format setting
        self.default_format_combo = QComboBox()
        self.default_format_combo.addItems(["CSV", "Excel"])
        settings_form.addRow("Default Output Format:", self.default_format_combo)

        # Autosave setting
        self.auto_save = QCheckBox("Auto-save Results")
        self.auto_save.setToolTip("Automatically save results after each simulation")
        settings_form.addRow("", self.auto_save)

        # File naming options
        self.file_naming_combo = QComboBox()
        self.file_naming_combo.addItems(["Timestamp", "Incremental Number", "Custom Prefix"])
        settings_form.addRow("File Naming Convention:", self.file_naming_combo)

        self.file_prefix_edit = QLineEdit("FLIPR_simulation")
        settings_form.addRow("Custom Prefix:", self.file_prefix_edit)

        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout()

        self.num_threads_spin = QSpinBox()
        self.num_threads_spin.setRange(1, 16)
        self.num_threads_spin.setValue(4)
        advanced_layout.addRow("Number of Threads:", self.num_threads_spin)


        advanced_group.setLayout(advanced_layout)

        # Assemble layout
        layout.addLayout(settings_form)
        layout.addWidget(advanced_group)
        layout.addStretch()

        # Save button
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self.save_application_settings)
        layout.addWidget(self.save_settings_btn)

        tab.setLayout(layout)

        # Load settings if they exist
        self.load_application_settings()

        return tab

    def browse_output_directory(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory",
                                              self.output_dir_edit.text())

        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def save_application_settings(self):
        """Save application settings to a file"""
        try:
            settings = {
                'output_directory': self.output_dir_edit.text(),
                'default_format': self.default_format_combo.currentText(),
                'auto_save': self.auto_save.isChecked(),
                'file_naming': self.file_naming_combo.currentText(),
                'file_prefix': self.file_prefix_edit.text(),
                'num_threads': self.num_threads_spin.value()
                # Removed random seed settings
            }

            # Create settings directory if it doesn't exist
            settings_dir = os.path.join(os.getcwd(), "settings")
            os.makedirs(settings_dir, exist_ok=True)

            # Save settings to JSON file
            settings_path = os.path.join(settings_dir, "app_settings.json")
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=4)

            self.debug_console.append_message(f"Settings saved to {settings_path}")
            self.statusBar().showMessage("Settings saved successfully", 3000)

        except Exception as e:
            self.debug_console.append_message(f"Error saving settings: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    def load_application_settings(self):
        """Load application settings from a file"""
        try:
            settings_path = os.path.join(os.getcwd(), "settings", "app_settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)

                # Apply settings to UI elements
                self.output_dir_edit.setText(settings.get('output_directory', os.path.join(os.getcwd(), "simulation_results")))

                format_index = self.default_format_combo.findText(settings.get('default_format', 'CSV'))
                if format_index >= 0:
                    self.default_format_combo.setCurrentIndex(format_index)

                self.auto_save.setChecked(settings.get('auto_save', False))

                naming_index = self.file_naming_combo.findText(settings.get('file_naming', 'Timestamp'))
                if naming_index >= 0:
                    self.file_naming_combo.setCurrentIndex(naming_index)

                self.file_prefix_edit.setText(settings.get('file_prefix', 'FLIPR_simulation'))
                self.num_threads_spin.setValue(settings.get('num_threads', 4))

                # Removed setting the random seed from here

                self.debug_console.append_message("Settings loaded successfully")
        except Exception as e:
            self.debug_console.append_message(f"Error loading settings: {str(e)}", level='WARNING')

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

    def run_error_comparison(self):
        """Run two simulations - one normal and one with errors - and compare the results"""
        try:
            # Get current configuration
            config = self.get_simulation_config()

            # Check if there are active errors
            if 'active_errors' not in config or not config['active_errors']:
                QMessageBox.warning(self, "No Errors Selected",
                                  "Please select at least one error type to run a comparison.")
                return

            # Create a copy of the config without errors
            normal_config = config.copy()
            if 'active_errors' in normal_config:
                del normal_config['active_errors']

            # Show status
            self.statusBar().showMessage("Running error comparison simulations...")
            self.debug_console.append_message("Starting error comparison simulation")

            # Run normal simulation
            self.debug_console.append_message("Running normal simulation...")
            normal_results = self.simulation_engine.simulate(normal_config)

            # Run simulation with errors
            self.debug_console.append_message("Running simulation with errors...")
            error_results = self.simulation_engine.simulate(config)

            # Create comparison plot
            self.debug_console.append_message("Creating comparison plot...")
            self.plot_error_comparison(normal_results, error_results)

            # Update status
            self.statusBar().showMessage("Error comparison completed")
            self.debug_console.append_message("Error comparison completed")

        except Exception as e:
            self.debug_console.append_message(f"Error running comparison: {str(e)}", level='ERROR')
            logger.error(f"Error running comparison: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to run comparison: {str(e)}")

    # Add a method to plot the error comparison
    def plot_error_comparison(self, normal_results, error_results):
        """Plot a comparison of normal vs error-affected results in a popup window"""
        try:
            # Create a new window for the comparison plot
            self.comparison_window = QDialog(self)
            self.comparison_window.setWindowTitle("Error Comparison")
            self.comparison_window.resize(900, 700)  # Set a reasonable size

            # Create layout for the window
            layout = QVBoxLayout()

            # Create matplotlib figure and canvas for the popup
            figure = Figure(figsize=(9, 7), dpi=100)
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, self.comparison_window)

            layout.addWidget(toolbar)
            layout.addWidget(canvas)

            # Add a close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.comparison_window.close)
            layout.addWidget(close_button)

            self.comparison_window.setLayout(layout)

            # Create a 2x2 grid of subplots
            gs = figure.add_gridspec(2, 2)

            # Top-left: Example well comparison (find a well with significant difference)
            ax1 = figure.add_subplot(gs[0, 0])

            # Find a well that shows significant difference
            diff_scores = []
            if normal_results and 'plate_data' in normal_results and 'plate_data' in error_results:
                normal_data = normal_results['plate_data']
                error_data = error_results['plate_data']

                for i in range(len(normal_data)):
                    if i < len(error_data):
                        # Calculate difference score (sum of squared differences)
                        diff = np.sum((normal_data[i] - error_data[i])**2)
                        diff_scores.append((i, diff))

                # Sort by difference score and pick the well with the most significant difference
                diff_scores.sort(key=lambda x: x[1], reverse=True)
                if diff_scores:
                    example_well_idx = diff_scores[0][0]
                    well_id = "Unknown"

                    # Get well ID from metadata
                    if 'metadata' in normal_results and example_well_idx < len(normal_results['metadata']):
                        well_id = normal_results['metadata'][example_well_idx].get('well_id', "Unknown")
                        cell_line = normal_results['metadata'][example_well_idx].get('cell_line', "Unknown")
                        agonist = normal_results['metadata'][example_well_idx].get('agonist', "Unknown")

                    # Plot this well for normal and error
                    time_points = normal_results['time_points']
                    ax1.plot(time_points, normal_data[example_well_idx], 'b-', label='Normal', linewidth=2)
                    ax1.plot(time_points, error_data[example_well_idx], 'r-', label='With Error', linewidth=2)

                    # Mark agonist addition time
                    if 'params' in normal_results and 'agonist_addition_time' in normal_results['params']:
                        agonist_time = normal_results['params']['agonist_addition_time']
                        ax1.axvline(x=agonist_time, color='k', linestyle='--')

                    ax1.set_title(f'Well {well_id}: Normal vs Error', fontsize=10)
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Fluorescence (A.U.)')
                    ax1.legend(fontsize=8)

            # Top-right: Box plot of peak responses
            ax2 = figure.add_subplot(gs[0, 1])

            # Calculate peak responses for both datasets
            normal_peaks = []
            error_peaks = []

            if 'params' in normal_results and 'agonist_addition_time' in normal_results['params']:
                agonist_time = normal_results['params']['agonist_addition_time']
                time_idx = int(agonist_time / normal_results['params']['time_interval'])
                normal_data = normal_results['plate_data']
                error_data = error_results['plate_data']

                for i in range(len(normal_data)):
                    if i < len(normal_results['metadata']) and i < len(error_data):
                        if normal_results['metadata'][i].get('valid', True):
                            # Calculate baseline and peak for normal
                            baseline = np.mean(normal_data[i][:time_idx]) if time_idx > 0 else normal_data[i][0]
                            peak = np.max(normal_data[i][time_idx:])
                            normal_peaks.append(peak - baseline)

                            # Calculate baseline and peak for error
                            baseline = np.mean(error_data[i][:time_idx]) if time_idx > 0 else error_data[i][0]
                            peak = np.max(error_data[i][time_idx:])
                            error_peaks.append(peak - baseline)

            # Create box plots
            if normal_peaks and error_peaks:
                box_data = [normal_peaks, error_peaks]
                ax2.boxplot(box_data, labels=['Normal', 'With Errors'])
                ax2.set_title('Peak Response Comparison', fontsize=10)
                ax2.set_ylabel('Peak Response (F-F0)')

                # Add percent difference annotation
                normal_mean = np.mean(normal_peaks)
                error_mean = np.mean(error_peaks)
                pct_diff = ((error_mean - normal_mean) / normal_mean) * 100

                ax2.annotate(f'Mean Difference: {pct_diff:.1f}%',
                           xy=(1.5, max(np.max(normal_peaks), np.max(error_peaks))),
                           xytext=(0, -20), textcoords='offset points',
                           ha='center', fontsize=8)

            # Bottom: All traces overlay
            ax3 = figure.add_subplot(gs[1, :])

            # Plot a subset of traces for clarity
            max_traces = 30  # Max number of traces to plot
            if 'plate_data' in normal_results and 'plate_data' in error_results:
                normal_data = normal_results['plate_data']
                error_data = error_results['plate_data']
                time_points = normal_results['time_points']

                step = max(1, len(normal_data) // max_traces)

                for i in range(0, len(normal_data), step):
                    if i < len(error_data):
                        ax3.plot(time_points, normal_data[i], 'b-', alpha=0.2, linewidth=0.5)
                        ax3.plot(time_points, error_data[i], 'r-', alpha=0.2, linewidth=0.5)

                # Add legend lines with higher opacity
                ax3.plot([], [], 'b-', label='Normal', linewidth=2)
                ax3.plot([], [], 'r-', label='With Errors', linewidth=2)

                # Mark agonist addition time
                if 'params' in normal_results and 'agonist_addition_time' in normal_results['params']:
                    agonist_time = normal_results['params']['agonist_addition_time']
                    ax3.axvline(x=agonist_time, color='k', linestyle='--')

                # Get active error types for title
                error_types = []
                if 'active_errors' in error_results.get('params', {}):
                    error_types = list(error_results['params']['active_errors'].keys())

                if error_types:
                    ax3.set_title(f'All Traces: {", ".join(error_types)}', fontsize=10)
                else:
                    ax3.set_title('All Traces: Normal vs With Errors', fontsize=10)

                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Fluorescence (A.U.)')
                ax3.legend(fontsize=8)

            # Add some information text
            figure.text(0.5, 0.01, 'Error comparison shows how selected error types affect calcium signals',
                       ha='center', fontsize=9)

            # Adjust layout
            figure.tight_layout()
            canvas.draw()

            # Show the window
            self.comparison_window.show()

            # Store for reference
            self.last_normal_results = normal_results
            self.last_error_results = error_results

        except Exception as e:
            self.debug_console.append_message(f"Error plotting comparison: {str(e)}", level='ERROR')
            logger.error(f"Error plotting comparison: {str(e)}", exc_info=True)


    def export_simulation_data(self):
        """Export simulation data to file"""
        if not hasattr(self, 'last_results') or not self.last_results:
            QMessageBox.warning(self, "No Data", "No simulation data available to export.")
            return

        try:
            # Get export format from settings
            export_format = self.default_format_combo.currentText() if hasattr(self, 'default_format_combo') else "CSV"

            # Generate filename based on settings
            file_naming = self.file_naming_combo.currentText() if hasattr(self, 'file_naming_combo') else "Timestamp"
            prefix = self.file_prefix_edit.text() if hasattr(self, 'file_prefix_edit') else "FLIPR_simulation"

            if file_naming == "Timestamp":
                filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            elif file_naming == "Incremental Number":
                # Find the next available number
                output_dir = self.output_dir_edit.text() if hasattr(self, 'output_dir_edit') else os.path.join(os.getcwd(), "simulation_results")
                existing_files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".csv" if export_format == "CSV" else ".xlsx")]
                next_num = 1
                for file in existing_files:
                    match = re.search(r'(\d+)', file)
                    if match:
                        num = int(match.group(1))
                        next_num = max(next_num, num + 1)
                filename = f"{prefix}_{next_num:03d}"
            else:  # Custom Prefix only
                filename = prefix

            # Set output directory
            output_dir = self.output_dir_edit.text() if hasattr(self, 'output_dir_edit') else os.path.join(os.getcwd(), "simulation_results")
            os.makedirs(output_dir, exist_ok=True)

            # Export based on format
            if export_format == "CSV":
                return self.export_to_csv(output_dir, filename)
            else:  # Excel
                return self.export_to_excel(output_dir, filename)

        except Exception as e:
            self.debug_console.append_message(f"Error exporting data: {str(e)}", level='ERROR')
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
            return False

    def export_to_csv(self, output_dir, filename_base):
        """Export simulation results to CSV files"""
        try:
            # Create main data file - traces for each well
            data_file = os.path.join(output_dir, f"{filename_base}_traces.csv")

            # Extract data
            plate_data = self.last_results['plate_data']
            time_points = self.last_results['time_points']
            metadata = self.last_results.get('metadata', [])

            # Create a DataFrame with time points as the first column
            df = pd.DataFrame()
            df['Time (s)'] = time_points

            # Add each well's data as a column
            for i, well_data in enumerate(plate_data):
                well_id = metadata[i].get('well_id', f"Well_{i+1}") if i < len(metadata) else f"Well_{i+1}"
                df[well_id] = well_data

            # Write to CSV
            df.to_csv(data_file, index=False)

            # Create metadata file
            meta_file = os.path.join(output_dir, f"{filename_base}_metadata.csv")

            if metadata:
                # Create DataFrame from metadata
                meta_df = pd.DataFrame(metadata)
                meta_df.to_csv(meta_file, index=False)

            # Create parameters file
            params_file = os.path.join(output_dir, f"{filename_base}_parameters.json")

            with open(params_file, 'w') as f:
                json.dump(self.last_results.get('params', {}), f, indent=4, default=str)

            self.debug_console.append_message(f"Data exported to {output_dir}")
            self.statusBar().showMessage(f"Data exported successfully to {output_dir}", 3000)
            return True

        except Exception as e:
            self.debug_console.append_message(f"Error exporting to CSV: {str(e)}", level='ERROR')
            raise e

    def export_to_excel(self, output_dir, filename_base):
        """Export simulation results to Excel file"""
        try:
            # Create Excel file
            excel_file = os.path.join(output_dir, f"{filename_base}.xlsx")

            # Create a Pandas Excel writer
            writer = pd.ExcelWriter(excel_file, engine='openpyxl')

            # Extract data
            plate_data = self.last_results['plate_data']
            time_points = self.last_results['time_points']
            metadata = self.last_results.get('metadata', [])

            # Create traces sheet
            traces_df = pd.DataFrame()
            traces_df['Time (s)'] = time_points

            for i, well_data in enumerate(plate_data):
                well_id = metadata[i].get('well_id', f"Well_{i+1}") if i < len(metadata) else f"Well_{i+1}"
                traces_df[well_id] = well_data

            traces_df.to_excel(writer, sheet_name='Traces', index=False)

            # Create metadata sheet
            if metadata:
                meta_df = pd.DataFrame(metadata)
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)

            # Create parameters sheet
            params = self.last_results.get('params', {})
            params_df = pd.DataFrame([(k, str(v)) for k, v in params.items()],
                                    columns=['Parameter', 'Value'])
            params_df.to_excel(writer, sheet_name='Parameters', index=False)

            # Save the Excel file
            writer.save()

            self.debug_console.append_message(f"Data exported to {excel_file}")
            self.statusBar().showMessage(f"Data exported successfully to {excel_file}", 3000)
            return True

        except Exception as e:
            self.debug_console.append_message(f"Error exporting to Excel: {str(e)}", level='ERROR')
            raise e


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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
