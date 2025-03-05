import os
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import csv
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
                            QListWidgetItem, QPushButton, QLabel, QLineEdit,
                            QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
                            QFormLayout, QGroupBox, QFileDialog, QMessageBox,
                            QTableWidget, QTableWidgetItem)

logger = logging.getLogger('FLIPR_Simulator.BatchProcessor')

class BatchWorker(QThread):
    """Worker thread for running batch simulations"""
    
    progress_updated = pyqtSignal(int, int)  # current, total
    simulation_completed = pyqtSignal(int, object)  # index, results
    batch_completed = pyqtSignal(list)  # list of result references
    error_occurred = pyqtSignal(int, str)  # index, error message
    
    def __init__(self, simulation_engine, batch_configs, output_dir):
        """Initialize the batch worker"""
        super().__init__()
        self.simulation_engine = simulation_engine
        self.batch_configs = batch_configs
        self.output_dir = output_dir
        self.results = []
        self.stop_requested = False
    
    def run(self):
        """Run the batch simulations"""
        try:
            total_sims = len(self.batch_configs)
            completed = 0
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            self.results = []
            
            # Run each simulation
            for i, config in enumerate(self.batch_configs):
                if self.stop_requested:
                    logger.info("Batch processing stopped by user")
                    break
                    
                try:
                    # Update progress
                    self.progress_updated.emit(i + 1, total_sims)
                    
                    # Run simulation
                    logger.info(f"Running batch simulation {i+1} of {total_sims}")
                    result = self.simulation_engine.simulate(config)
                    
                    # Save individual result
                    result_path = self.save_simulation_result(i, config, result)
                    
                    # Add to results list
                    self.results.append({
                        'index': i,
                        'config': config,
                        'result_path': result_path,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Signal completion of this simulation
                    self.simulation_completed.emit(i, result)
                    completed += 1
                    
                except Exception as e:
                    logger.error(f"Error in batch simulation {i+1}: {str(e)}", exc_info=True)
                    self.error_occurred.emit(i, str(e))
            
            # Save batch summary
            self.save_batch_summary()
            
            # Signal batch completion
            self.batch_completed.emit(self.results)
            logger.info(f"Batch processing completed: {completed} of {total_sims} simulations successful")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
            self.error_occurred.emit(-1, str(e))
    
    def save_simulation_result(self, index, config, result):
        """Save an individual simulation result"""
        try:
            # Create directory for this simulation
            sim_dir = os.path.join(self.output_dir, f"simulation_{index+1:03d}")
            os.makedirs(sim_dir, exist_ok=True)
            
            # Save configuration
            config_path = os.path.join(sim_dir, "config.json")
            with open(config_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_config = {}
                for key, value in config.items():
                    if isinstance(value, np.ndarray):
                        serializable_config[key] = value.tolist()
                    else:
                        serializable_config[key] = value
                        
                json.dump(serializable_config, f, indent=4)
            
            # Save raw data as CSV
            data_path = os.path.join(sim_dir, "plate_data.csv")
            rows, cols = 8, 12  # Assuming 96-well plate
            
            with open(data_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write time points as header
                header = ['Well'] + [f"{t:.1f}" for t in result['time_points']]
                writer.writerow(header)
                
                # Write data for each well
                for i, well_data in enumerate(result['plate_data']):
                    row, col = i // cols, i % cols
                    well_id = f"{chr(65 + row)}{col + 1}"
                    writer.writerow([well_id] + list(well_data))
            
            # Save metadata
            meta_path = os.path.join(sim_dir, "metadata.csv")
            with open(meta_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['well_id', 'cell_id', 'cell_line', 'agonist', 'valid'])
                
                for meta in result['metadata']:
                    writer.writerow([
                        meta.get('well_id', ''),
                        meta.get('cell_id', ''),
                        meta.get('cell_line', ''),
                        meta.get('agonist', ''),
                        meta.get('valid', True)
                    ])
            
            # Save a summary plot
            plot_path = os.path.join(sim_dir, "summary_plot.png")
            self.create_summary_plot(result, plot_path)
            
            return sim_dir
            
        except Exception as e:
            logger.error(f"Error saving simulation {index+1} result: {str(e)}", exc_info=True)
            return None
    
    def save_batch_summary(self):
        """Save a summary of the batch run"""
        try:
            # Create summary file
            summary_path = os.path.join(self.output_dir, "batch_summary.csv")
            
            with open(summary_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Simulation Index', 'Timestamp', 'Error Type', 'Result Path'])
                
                for result in self.results:
                    # Get error type if any
                    error_type = "None"
                    if 'config' in result and 'active_errors' in result['config']:
                        active_errors = []
                        for error_name, settings in result['config']['active_errors'].items():
                            if settings.get('active', False):
                                active_errors.append(error_name)
                        
                        if active_errors:
                            error_type = ", ".join(active_errors)
                    
                    writer.writerow([
                        result.get('index', -1) + 1,
                        result.get('timestamp', ''),
                        error_type,
                        result.get('result_path', '')
                    ])
            
            logger.info(f"Batch summary saved to {summary_path}")
            return summary_path
            
        except Exception as e:
            logger.error(f"Error saving batch summary: {str(e)}", exc_info=True)
            return None
    
    def create_summary_plot(self, result, output_path):
        """Create a summary plot for a simulation result"""
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot all traces
            plate_data = result['plate_data']
            time_points = result['time_points']
            metadata = result['metadata']
            
            # Group by cell line for coloring
            cell_line_colors = {
                'Positive Control': 'red',
                'Negative Control': 'blue',
                'Neurotypical': 'green',
                'ASD': 'purple',
                'FXS': 'orange'
            }
            
            # Plot each well trace
            for i, well_data in enumerate(plate_data):
                if i < len(metadata) and metadata[i].get('valid', True):
                    cell_line = metadata[i].get('cell_line', 'Unknown')
                    color = cell_line_colors.get(cell_line, 'gray')
                    ax1.plot(time_points, well_data, color=color, alpha=0.3, linewidth=0.5)
            
            # Add legend entries
            for cell_line, color in cell_line_colors.items():
                ax1.plot([], [], color=color, label=cell_line, linewidth=2)
            
            ax1.set_title('All Traces')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Fluorescence (A.U.)')
            ax1.legend()
            
            # Create a heatmap of peak responses
            heatmap_data = np.zeros((8, 12))  # Assuming 96-well plate
            
            # Calculate peak responses
            if 'params' in result:
                baseline_end = int(result['params'].get('agonist_addition_time', 10) / 
                                  result['params'].get('time_interval', 0.4))
                
                for i, well_data in enumerate(plate_data):
                    row, col = i // 12, i % 12
                    if row < 8 and col < 12:
                        if len(well_data) > baseline_end:
                            baseline = np.mean(well_data[:baseline_end]) if baseline_end > 0 else well_data[0]
                            peak = np.max(well_data[baseline_end:])
                            heatmap_data[row, col] = peak - baseline
            
            # Create heatmap
            im = ax2.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
            plt.colorbar(im, ax=ax2, label='Peak Response (F-F0)')
            
            # Add well labels
            for i in range(8):
                for j in range(12):
                    ax2.text(j, i, f"{chr(65 + i)}{j + 1}", ha='center', va='center', 
                            color='white', fontsize=8)
            
            ax2.set_title('Peak Response Heatmap')
            ax2.set_xticks(range(12))
            ax2.set_yticks(range(8))
            ax2.set_xticklabels(range(1, 13))
            ax2.set_yticklabels('ABCDEFGH')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            
            logger.info(f"Summary plot saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating summary plot: {str(e)}", exc_info=True)
    
    def stop(self):
        """Request stopping the batch processing"""
        self.stop_requested = True


class BatchProcessor(QWidget):
    """Widget for configuring and running batch simulations"""
    
    def __init__(self, simulation_engine, config_manager, parent=None):
        """Initialize the batch processor"""
        super().__init__(parent)
        self.simulation_engine = simulation_engine
        self.config_manager = config_manager
        
        # Batch configurations
        self.batch_configs = []
        
        # Output directory
        self.output_dir = os.path.join(os.getcwd(), 'batch_results', 
                                      datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        # Batch configuration section
        config_group = QGroupBox("Batch Configuration")
        config_layout = QVBoxLayout()
        
        # Batch list
        list_layout = QHBoxLayout()
        
        self.batch_list = QListWidget()
        self.batch_list.setMinimumHeight(200)
        
        # List controls
        list_controls = QVBoxLayout()
        self.add_config_btn = QPushButton("Add Configuration")
        self.add_config_btn.clicked.connect(self.add_configuration)
        
        self.duplicate_config_btn = QPushButton("Duplicate")
        self.duplicate_config_btn.clicked.connect(self.duplicate_configuration)
        
        self.edit_config_btn = QPushButton("Edit")
        self.edit_config_btn.clicked.connect(self.edit_configuration)
        
        self.remove_config_btn = QPushButton("Remove")
        self.remove_config_btn.clicked.connect(self.remove_configuration)
        
        list_controls.addWidget(self.add_config_btn)
        list_controls.addWidget(self.duplicate_config_btn)
        list_controls.addWidget(self.edit_config_btn)
        list_controls.addWidget(self.remove_config_btn)
        list_controls.addStretch()
        
        list_layout.addWidget(self.batch_list, 3)
        list_layout.addLayout(list_controls, 1)
        
        config_layout.addLayout(list_layout)
        
        # Preset error scenarios
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Add Preset Error Scenario:"))
        
        self.preset_combo = QComboBox()
        preset_layout.addWidget(self.preset_combo, 2)
        
        self.add_preset_btn = QPushButton("Add to Batch")
        self.add_preset_btn.clicked.connect(self.add_preset_to_batch)
        preset_layout.addWidget(self.add_preset_btn)
        
        config_layout.addLayout(preset_layout)
        
        # Load presets
        self.load_preset_scenarios()
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Batch execution settings
        execution_group = QGroupBox("Execution Settings")
        execution_layout = QFormLayout()
        
        self.output_dir_edit = QLineEdit(self.output_dir)
        self.output_dir_edit.setReadOnly(True)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(browse_btn)
        
        execution_layout.addRow("Output Directory:", output_dir_layout)
        
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100)
        self.iterations_spin.setValue(1)
        execution_layout.addRow("Iterations per Configuration:", self.iterations_spin)
        
        self.random_seed_check = QCheckBox("Use Random Seed")
        self.random_seed_check.setChecked(True)
        execution_layout.addRow("", self.random_seed_check)
        
        execution_group.setLayout(execution_layout)
        main_layout.addWidget(execution_group)
        
        # Progress section
        progress_group = QGroupBox("Batch Progress")
        progress_layout = QVBoxLayout()
        
        # Progress display
        self.progress_label = QLabel("Ready to start batch processing")
        progress_layout.addWidget(self.progress_label)
        
        # Results table
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Index", "Configuration", "Status", "Result Path"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        progress_layout.addWidget(self.results_table)
        
        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.run_batch_btn = QPushButton("Run Batch")
        self.run_batch_btn.clicked.connect(self.run_batch)
        
        self.stop_batch_btn = QPushButton("Stop")
        self.stop_batch_btn.setEnabled(False)
        self.stop_batch_btn.clicked.connect(self.stop_batch)
        
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.setEnabled(False)
        self.export_results_btn.clicked.connect(self.export_results)
        
        controls_layout.addWidget(self.run_batch_btn)
        controls_layout.addWidget(self.stop_batch_btn)
        controls_layout.addWidget(self.export_results_btn)
        
        main_layout.addLayout(controls_layout)
        
        self.setLayout(main_layout)
    
    def load_preset_scenarios(self):
        """Load preset error scenarios into the combo box"""
        try:
            # Clear combo box
            self.preset_combo.clear()
            
            # Add "Normal" scenario
            self.preset_combo.addItem("Normal (No Errors)")
            
            # Get presets from config manager
            presets = self.config_manager.list_presets()
            
            # Add error scenario presets
            for preset in presets:
                if preset.get('category') == 'error_scenario':
                    name = preset.get('name', '')
                    desc = preset.get('description', '')
                    self.preset_combo.addItem(f"{name} - {desc}", name)
            
            logger.info(f"Loaded {self.preset_combo.count()} preset scenarios")
            
        except Exception as e:
            logger.error(f"Error loading preset scenarios: {str(e)}", exc_info=True)
    
    def add_preset_to_batch(self):
        """Add the selected preset to the batch list"""
        try:
            preset_text = self.preset_combo.currentText()
            preset_name = self.preset_combo.currentData()
            
            if preset_text == "Normal (No Errors)":
                # Create a configuration with no errors
                config = self.config_manager.get_simulation_config()
                config['name'] = "Normal Simulation"
                
                # Add to batch list
                self.add_config_to_batch(config)
                
            elif preset_name:
                # Load the preset
                preset_config = self.config_manager.load_preset(f"{preset_name}.json")
                
                if preset_config:
                    # Get simulation config
                    config = self.config_manager.get_simulation_config()
                    config['name'] = preset_name
                    
                    # Set active errors from preset
                    if 'errors' in preset_config:
                        # Convert to active_errors format
                        active_errors = {}
                        for error_name, settings in preset_config['errors'].items():
                            if settings.get('active', False):
                                active_errors[error_name] = settings
                        
                        config['active_errors'] = active_errors
                    
                    # Add to batch list
                    self.add_config_to_batch(config)
                    
                else:
                    QMessageBox.warning(self, "Warning", f"Failed to load preset: {preset_name}")
            
        except Exception as e:
            logger.error(f"Error adding preset to batch: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add preset: {str(e)}")
    
    def add_config_to_batch(self, config):
        """Add a configuration to the batch list"""
        # Add configuration to the list
        self.batch_configs.append(config)
        
        # Add to the list widget
        item_text = config.get('name', f"Configuration {len(self.batch_configs)}")
        if 'active_errors' in config:
            error_names = list(config['active_errors'].keys())
            if error_names:
                item_text += f" ({', '.join(error_names)})"
        
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, len(self.batch_configs) - 1)  # Store index
        self.batch_list.addItem(item)
        
        logger.info(f"Added configuration to batch: {item_text}")
    
    def add_configuration(self):
        """Add a new configuration to the batch"""
        try:
            # Get base configuration
            config = self.config_manager.get_simulation_config()
            config['name'] = f"Configuration {len(self.batch_configs) + 1}"
            
            # Add to batch
            self.add_config_to_batch(config)
            
        except Exception as e:
            logger.error(f"Error adding configuration: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add configuration: {str(e)}")
    
    def duplicate_configuration(self):
        """Duplicate the selected configuration"""
        try:
            selected_items = self.batch_list.selectedItems()
            
            if not selected_items:
                QMessageBox.warning(self, "Warning", "Please select a configuration to duplicate.")
                return
            
            # Get the selected configuration
            index = selected_items[0].data(Qt.UserRole)
            if index < 0 or index >= len(self.batch_configs):
                return
                
            # Duplicate configuration
            config = self.batch_configs[index].copy()
            config['name'] = f"{config.get('name', 'Configuration')} (Copy)"
            
            # Add to batch
            self.add_config_to_batch(config)
            
        except Exception as e:
            logger.error(f"Error duplicating configuration: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to duplicate configuration: {str(e)}")
    
    def edit_configuration(self):
        """Edit the selected configuration"""
        # TODO: Implement configuration editor dialog
        QMessageBox.information(self, "Not Implemented", 
                             "Configuration editing will be implemented in a future version.")
    
    def remove_configuration(self):
        """Remove the selected configuration from the batch"""
        try:
            selected_items = self.batch_list.selectedItems()
            
            if not selected_items:
                QMessageBox.warning(self, "Warning", "Please select a configuration to remove.")
                return
            
            # Get the selected configuration
            item = selected_items[0]
            index = item.data(Qt.UserRole)
            
            if index < 0 or index >= len(self.batch_configs):
                return
                
            # Confirm removal
            confirm = QMessageBox.question(
                self, "Confirm Removal", 
                f"Are you sure you want to remove '{item.text()}' from the batch?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if confirm == QMessageBox.Yes:
                # Remove from batch configs and list widget
                self.batch_configs.pop(index)
                self.batch_list.takeItem(self.batch_list.row(item))
                
                # Update indices in list widget
                for i in range(self.batch_list.count()):
                    list_item = self.batch_list.item(i)
                    item_index = list_item.data(Qt.UserRole)
                    
                    if item_index > index:
                        list_item.setData(Qt.UserRole, item_index - 1)
                
                logger.info(f"Removed configuration {index+1} from batch")
            
        except Exception as e:
            logger.error(f"Error removing configuration: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to remove configuration: {str(e)}")
    
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", 
                                                  self.output_dir)
        
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_edit.setText(dir_path)
    
    def run_batch(self):
        """Run the batch simulations"""
        try:
            if not self.batch_configs:
                QMessageBox.warning(self, "Warning", "Please add at least one configuration to the batch.")
                return
            
            # Get batch settings
            iterations = self.iterations_spin.value()
            use_random_seed = self.random_seed_check.isChecked()
            
            # Prepare expanded configurations
            expanded_configs = []
            
            for config in self.batch_configs:
                for i in range(iterations):
                    # Create a copy of the configuration
                    config_copy = config.copy()
                    
                    # Set different random seed for each iteration if enabled
                    if use_random_seed:
                        config_copy['random_seed'] = np.random.randint(1, 10000)
                    
                    # Add iteration information
                    if iterations > 1:
                        config_copy['iteration'] = i + 1
                        config_copy['name'] = f"{config.get('name', 'Configuration')} (Iteration {i+1})"
                    
                    expanded_configs.append(config_copy)
            
            # Set up results table
            self.results_table.setRowCount(len(expanded_configs))
            for i, config in enumerate(expanded_configs):
                self.results_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.results_table.setItem(i, 1, QTableWidgetItem(config.get('name', f"Configuration {i+1}")))
                self.results_table.setItem(i, 2, QTableWidgetItem("Pending"))
                self.results_table.setItem(i, 3, QTableWidgetItem(""))
            
            # Update UI
            self.run_batch_btn.setEnabled(False)
            self.stop_batch_btn.setEnabled(True)
            self.export_results_btn.setEnabled(False)
            self.progress_label.setText(f"Running batch: 0/{len(expanded_configs)} completed")
            
            # Create and start worker thread
            self.batch_worker = BatchWorker(self.simulation_engine, expanded_configs, self.output_dir)
            self.batch_worker.progress_updated.connect(self.update_progress)
            self.batch_worker.simulation_completed.connect(self.simulation_completed)
            self.batch_worker.batch_completed.connect(self.batch_completed)
            self.batch_worker.error_occurred.connect(self.batch_error)
            
            self.batch_worker.start()
            
            logger.info(f"Started batch processing with {len(expanded_configs)} simulations")
            
        except Exception as e:
            logger.error(f"Error starting batch processing: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to start batch processing: {str(e)}")
    
    def update_progress(self, current, total):
        """Update progress display"""
        self.progress_label.setText(f"Running batch: {current}/{total} completed")
    
    def simulation_completed(self, index, result):
        """Handle completion of a single simulation"""
        try:
            # Update status in the table
            if index < self.results_table.rowCount():
                self.results_table.setItem(index, 2, QTableWidgetItem("Completed"))
                
                # Add result path if available
                if hasattr(self.batch_worker, 'results') and index < len(self.batch_worker.results):
                    result_path = self.batch_worker.results[index].get('result_path', '')
                    self.results_table.setItem(index, 3, QTableWidgetItem(result_path))
            
        except Exception as e:
            logger.error(f"Error updating simulation completion: {str(e)}", exc_info=True)
    
    def batch_completed(self, results):
        """Handle completion of the entire batch"""
        # Update UI
        self.run_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.export_results_btn.setEnabled(True)
        
        total = self.results_table.rowCount()
        completed = sum(1 for i in range(total) 
                       if self.results_table.item(i, 2).text() == "Completed")
        
        self.progress_label.setText(f"Batch processing complete: {completed}/{total} simulations successful")
        
        # Show completion message
        QMessageBox.information(self, "Batch Complete", 
                             f"Batch processing completed.\n\n"
                             f"{completed} of {total} simulations successful.\n\n"
                             f"Results saved to: {self.output_dir}")
        
        logger.info(f"Batch processing completed: {completed}/{total} simulations successful")
    
    def batch_error(self, index, error_msg):
        """Handle batch processing error"""
        if index >= 0 and index < self.results_table.rowCount():
            # Update status for the specific simulation
            self.results_table.setItem(index, 2, QTableWidgetItem(f"Error: {error_msg}"))
        else:
            # General batch error
            self.progress_label.setText(f"Batch processing error: {error_msg}")
            
            # Reset UI
            self.run_batch_btn.setEnabled(True)
            self.stop_batch_btn.setEnabled(False)
            
            QMessageBox.critical(self, "Batch Error", f"Error in batch processing: {error_msg}")
            
        logger.error(f"Batch error (index {index}): {error_msg}")
    
    def stop_batch(self):
        """Stop the batch processing"""
        if hasattr(self, 'batch_worker') and self.batch_worker.isRunning():
            # Request stop
            self.batch_worker.stop()
            self.stop_batch_btn.setEnabled(False)
            self.progress_label.setText("Stopping batch processing...")
            
            logger.info("Batch processing stop requested")
    
    def export_results(self):
        """Export batch results to a selected location"""
        # Select export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory", 
                                                   os.getcwd())
        
        if not export_dir:
            return
            
        try:
            # Export batch summary
            if hasattr(self, 'batch_worker') and hasattr(self.batch_worker, 'results'):
                # Create summary file
                summary_path = os.path.join(export_dir, "batch_summary.csv")
                
                with open(summary_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Simulation Index', 'Configuration', 'Status', 'Timestamp', 'Result Path'])
                    
                    for i in range(self.results_table.rowCount()):
                        index_item = self.results_table.item(i, 0)
                        config_item = self.results_table.item(i, 1)
                        status_item = self.results_table.item(i, 2)
                        path_item = self.results_table.item(i, 3)
                        
                        # Get result data if available
                        timestamp = ""
                        if i < len(self.batch_worker.results):
                            timestamp = self.batch_worker.results[i].get('timestamp', '')
                        
                        writer.writerow([
                            index_item.text() if index_item else "",
                            config_item.text() if config_item else "",
                            status_item.text() if status_item else "",
                            timestamp,
                            path_item.text() if path_item else ""
                        ])
                
                # Copy batch results to export directory
                if QMessageBox.question(
                    self, "Export Results", 
                    f"Do you want to copy all result files to {export_dir}?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                ) == QMessageBox.Yes:
                    # Create results directory
                    export_results_dir = os.path.join(export_dir, "batch_results")
                    os.makedirs(export_results_dir, exist_ok=True)
                    
                    # Copy result folders
                    import shutil
                    for result in self.batch_worker.results:
                        result_path = result.get('result_path')
                        if result_path and os.path.exists(result_path):
                            # Get simulation index
                            sim_index = result.get('index', 0)
                            
                            # Create destination folder
                            dest_folder = os.path.join(export_results_dir, f"simulation_{sim_index+1:03d}")
                            
                            # Copy folder
                            if os.path.exists(dest_folder):
                                shutil.rmtree(dest_folder)
                            shutil.copytree(result_path, dest_folder)
                    
                QMessageBox.information(self, "Export Complete", 
                                     f"Batch results exported to {export_dir}")
                
                logger.info(f"Batch results exported to {export_dir}")
                
            else:
                QMessageBox.warning(self, "Warning", "No batch results available to export.")
                
        except Exception as e:
            logger.error(f"Error exporting batch results: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to export batch results: {str(e)}")
