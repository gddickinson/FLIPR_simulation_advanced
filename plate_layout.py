import numpy as np
import pandas as pd
import logging
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QLabel, QComboBox, QTableWidget, QTableWidgetItem,
                           QPushButton, QHeaderView, QFileDialog, QMessageBox,
                           QGroupBox, QFormLayout, QLineEdit, QDialog, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

logger = logging.getLogger('FLIPR_Simulator.PlateLayoutEditor')

class PlateLayoutEditor(QWidget):
    """Widget for editing plate layouts (cell lines, agonists, etc.)"""

    # Signal emitted when layout changes
    layout_changed = pyqtSignal(str, object)  # layout_type, new_layout

    def __init__(self, config_manager, parent=None):
        """Initialize the plate layout editor"""
        super().__init__(parent)
        self.config_manager = config_manager

        # Current plate format
        self.plate_format = '96-well'
        self.rows = 8
        self.cols = 12

        # Current layouts
        self.agonist_layout = None
        self.cell_line_layout = None
        self.cell_id_layout = None
        self.group_id_layout = None  # New: Add group ID layout

        # Initialize UI
        self.init_ui()

        # Load default layouts
        self.load_default_layouts()

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()

        # Plate format selector
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Plate Format:"))

        self.format_combo = QComboBox()
        self.format_combo.addItems(["96-well", "384-well"])
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        format_layout.addWidget(self.format_combo)

        self.fill_pattern_combo = QComboBox()
        self.fill_pattern_combo.addItems(["Checkerboard", "By Row", "By Column", "All Same"])
        format_layout.addWidget(QLabel("Fill Pattern:"))
        format_layout.addWidget(self.fill_pattern_combo)

        format_layout.addStretch()
        main_layout.addLayout(format_layout)

        # Tab layout for different layout editors
        tab_layout = QHBoxLayout()

        # Layout editors
        cell_line_group = QGroupBox("Cell Line Layout")
        self.cell_line_editor = self.create_layout_editor("cell_line")
        cell_line_layout = QVBoxLayout()
        cell_line_layout.addWidget(self.cell_line_editor)
        cell_line_group.setLayout(cell_line_layout)

        agonist_group = QGroupBox("Agonist Layout")
        self.agonist_editor = self.create_layout_editor("agonist")
        agonist_layout = QVBoxLayout()
        agonist_layout.addWidget(self.agonist_editor)
        agonist_group.setLayout(agonist_layout)

        # New: Add Group ID layout editor
        group_id_group = QGroupBox("Group ID Layout")
        self.group_id_editor = self.create_layout_editor("group_id")
        group_id_layout = QVBoxLayout()
        group_id_layout.addWidget(self.group_id_editor)
        group_id_group.setLayout(group_id_layout)

        tab_layout.addWidget(cell_line_group)
        tab_layout.addWidget(agonist_group)
        tab_layout.addWidget(group_id_group)  # Add the new group ID editor

        main_layout.addLayout(tab_layout)

        # Cell and agonist management
        management_layout = QHBoxLayout()

        # Cell line management
        cell_mgmt_group = QGroupBox("Cell Line Management")
        cell_mgmt_layout = QVBoxLayout()

        # Cell line list
        self.cell_line_list = QTableWidget(0, 5)
        self.cell_line_list.setHorizontalHeaderLabels([
            "Cell Line", "Baseline", "Peak (Ionomycin)", "Peak (Other)", "Rise Rate", "Decay Rate"
        ])
        self.cell_line_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        cell_mgmt_layout.addWidget(self.cell_line_list)

        # Cell line controls
        cell_controls = QHBoxLayout()
        self.add_cell_btn = QPushButton("Add Cell Line")
        self.add_cell_btn.clicked.connect(self.add_cell_line)
        self.edit_cell_btn = QPushButton("Edit")
        self.edit_cell_btn.clicked.connect(self.edit_cell_line)
        self.remove_cell_btn = QPushButton("Remove")
        self.remove_cell_btn.clicked.connect(self.remove_cell_line)

        cell_controls.addWidget(self.add_cell_btn)
        cell_controls.addWidget(self.edit_cell_btn)
        cell_controls.addWidget(self.remove_cell_btn)
        cell_mgmt_layout.addLayout(cell_controls)

        cell_mgmt_group.setLayout(cell_mgmt_layout)

        # Agonist management
        agonist_mgmt_group = QGroupBox("Agonist Management")
        agonist_mgmt_layout = QVBoxLayout()

        # Agonist list
        self.agonist_list = QTableWidget(0, 2)
        self.agonist_list.setHorizontalHeaderLabels(["Agonist", "Response Factor"])
        self.agonist_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        agonist_mgmt_layout.addWidget(self.agonist_list)

        # Agonist controls
        agonist_controls = QHBoxLayout()
        self.add_agonist_btn = QPushButton("Add Agonist")
        self.add_agonist_btn.clicked.connect(self.add_agonist)
        self.edit_agonist_btn = QPushButton("Edit")
        self.edit_agonist_btn.clicked.connect(self.edit_agonist)
        self.remove_agonist_btn = QPushButton("Remove")
        self.remove_agonist_btn.clicked.connect(self.remove_agonist)

        agonist_controls.addWidget(self.add_agonist_btn)
        agonist_controls.addWidget(self.edit_agonist_btn)
        agonist_controls.addWidget(self.remove_agonist_btn)
        agonist_mgmt_layout.addLayout(agonist_controls)

        agonist_mgmt_group.setLayout(agonist_mgmt_layout)

        management_layout.addWidget(cell_mgmt_group)
        management_layout.addWidget(agonist_mgmt_group)

        main_layout.addLayout(management_layout)

        # Save/load controls
        control_layout = QHBoxLayout()

        self.save_layout_btn = QPushButton("Save Layouts")
        self.save_layout_btn.clicked.connect(self.save_layouts)

        self.load_layout_btn = QPushButton("Load Layouts")
        self.load_layout_btn.clicked.connect(self.load_layouts)

        self.reset_layout_btn = QPushButton("Reset to Default")
        self.reset_layout_btn.clicked.connect(self.load_default_layouts)

        control_layout.addWidget(self.save_layout_btn)
        control_layout.addWidget(self.load_layout_btn)
        control_layout.addWidget(self.reset_layout_btn)

        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)

    def create_layout_editor(self, layout_type):
        """Create a table widget for editing plate layouts"""
        table = QTableWidget(self.rows, self.cols)

        # Set row and column headers
        row_labels = [chr(65 + i) for i in range(self.rows)]
        col_labels = [str(i+1) for i in range(self.cols)]

        table.setVerticalHeaderLabels(row_labels)
        table.setHorizontalHeaderLabels(col_labels)

        # Set cell size
        for i in range(self.rows):
            table.setRowHeight(i, 30)
        for i in range(self.cols):
            table.setColumnWidth(i, 60)

        # Connect signals
        table.cellChanged.connect(lambda row, col: self.on_cell_changed(layout_type, row, col))

        return table

    def on_format_changed(self, format_text):
        """Handle changes to plate format"""
        self.plate_format = format_text

        if format_text == '96-well':
            self.rows = 8
            self.cols = 12
        elif format_text == '384-well':
            self.rows = 16
            self.cols = 24

        # Update layout editors
        self.update_layout_tables()

        # Reset layouts
        self.load_default_layouts()

    def update_layout_tables(self):
        """Update layout editor tables for new plate format"""
        # Update cell line editor
        self.cell_line_editor.setRowCount(self.rows)
        self.cell_line_editor.setColumnCount(self.cols)

        # Update agonist editor
        self.agonist_editor.setRowCount(self.rows)
        self.agonist_editor.setColumnCount(self.cols)

        # Update group ID editor
        self.group_id_editor.setRowCount(self.rows)
        self.group_id_editor.setColumnCount(self.cols)

        # Set row and column headers
        row_labels = [chr(65 + i) for i in range(self.rows)]
        col_labels = [str(i+1) for i in range(self.cols)]

        self.cell_line_editor.setVerticalHeaderLabels(row_labels)
        self.cell_line_editor.setHorizontalHeaderLabels(col_labels)

        self.agonist_editor.setVerticalHeaderLabels(row_labels)
        self.agonist_editor.setHorizontalHeaderLabels(col_labels)

        self.group_id_editor.setVerticalHeaderLabels(row_labels)
        self.group_id_editor.setHorizontalHeaderLabels(col_labels)

        # Adjust cell sizes
        for i in range(self.rows):
            self.cell_line_editor.setRowHeight(i, 30)
            self.agonist_editor.setRowHeight(i, 30)
            self.group_id_editor.setRowHeight(i, 30)

        for i in range(self.cols):
            width = 60 if self.cols <= 12 else 40
            self.cell_line_editor.setColumnWidth(i, width)
            self.agonist_editor.setColumnWidth(i, width)
            self.group_id_editor.setColumnWidth(i, width)

    def create_default_group_id_layout(self, rows, cols, default_group='Group A'):
        """Create a default group ID layout"""
        layout = np.empty((rows, cols), dtype=object)

        # Initialize with default group
        layout.fill(default_group)

        # Create a pattern with different groups
        group_names = ['Group A', 'Group B', 'Group C', 'Group D']

        # Assign groups by columns
        for j in range(cols):
            group_idx = j % len(group_names)
            for i in range(rows):
                layout[i, j] = group_names[group_idx]

        return layout


    def on_cell_changed(self, layout_type, row, col):
        """Handle changes to layout cells"""
        if layout_type == 'cell_line':
            if self.cell_line_layout is not None:
                self.cell_line_layout[row, col] = self.cell_line_editor.item(row, col).text()
                self.layout_changed.emit('cell_line', self.cell_line_layout)

        elif layout_type == 'agonist':
            if self.agonist_layout is not None:
                self.agonist_layout[row, col] = self.agonist_editor.item(row, col).text()
                self.layout_changed.emit('agonist', self.agonist_layout)

        # New: Handle group_id layout changes
        elif layout_type == 'group_id':
            if self.group_id_layout is not None:
                self.group_id_layout[row, col] = self.group_id_editor.item(row, col).text()
                self.layout_changed.emit('group_id', self.group_id_layout)

    def populate_layout_table(self, table, layout):
        """Populate a layout editor table with values"""
        if layout is None:
            return

        for i in range(min(self.rows, layout.shape[0])):
            for j in range(min(self.cols, layout.shape[1])):
                value = layout[i, j]
                item = QTableWidgetItem(str(value))
                table.setItem(i, j, item)

    def load_default_layouts(self):
        """Load default plate layouts"""
        # Create default layouts
        if self.plate_format == '96-well':
            self.agonist_layout = self.create_default_agonist_layout(8, 12)
            self.cell_line_layout = self.create_default_cell_line_layout(8, 12)
            self.cell_id_layout = self.create_default_cell_id_layout(8, 12)
            self.group_id_layout = self.create_default_group_id_layout(8, 12)  # New
        else:  # 384-well
            self.agonist_layout = self.create_default_agonist_layout(16, 24)
            self.cell_line_layout = self.create_default_cell_line_layout(16, 24)
            self.cell_id_layout = self.create_default_cell_id_layout(16, 24)
            self.group_id_layout = self.create_default_group_id_layout(16, 24)  # New

        # Update layout editors
        self.populate_layout_table(self.cell_line_editor, self.cell_line_layout)
        self.populate_layout_table(self.agonist_editor, self.agonist_layout)
        self.populate_layout_table(self.group_id_editor, self.group_id_layout)  # New

        # Update cell line and agonist lists
        self.populate_cell_line_list()
        self.populate_agonist_list()

        # Emit change signals
        self.layout_changed.emit('cell_line', self.cell_line_layout)
        self.layout_changed.emit('agonist', self.agonist_layout)
        self.layout_changed.emit('group_id', self.group_id_layout)  # New

    def create_default_agonist_layout(self, rows, cols):
        """Create a default agonist layout"""
        layout = np.empty((rows, cols), dtype=object)

        # Default pattern: ATP in columns 1-6, Ionomycin in 7-10, Buffer in 11-12
        for i in range(rows):
            for j in range(cols):
                if self.plate_format == '96-well':
                    if j < 6:
                        layout[i, j] = 'ATP'
                    elif j < 10:
                        layout[i, j] = 'Ionomycin'
                    else:
                        layout[i, j] = 'Buffer'
                else:  # 384-well
                    if j < 12:
                        layout[i, j] = 'ATP'
                    elif j < 20:
                        layout[i, j] = 'Ionomycin'
                    else:
                        layout[i, j] = 'Buffer'

        return layout

    def create_default_cell_line_layout(self, rows, cols):
        """Create a default cell line layout"""
        layout = np.empty((rows, cols), dtype=object)

        # Set up some default patterns
        for i in range(rows):
            for j in range(cols):
                if self.plate_format == '96-well':
                    if j == cols - 1:
                        layout[i, j] = 'Positive Control'
                    elif j == cols - 2:
                        layout[i, j] = 'Negative Control'
                    elif j % 3 == 0:
                        layout[i, j] = 'Neurotypical'
                    elif j % 3 == 1:
                        layout[i, j] = 'ASD'
                    else:
                        layout[i, j] = 'FXS'
                else:  # 384-well
                    if j == cols - 1 or j == cols - 2:
                        layout[i, j] = 'Positive Control'
                    elif j == cols - 3 or j == cols - 4:
                        layout[i, j] = 'Negative Control'
                    elif j % 6 < 2:
                        layout[i, j] = 'Neurotypical'
                    elif j % 6 < 4:
                        layout[i, j] = 'ASD'
                    else:
                        layout[i, j] = 'FXS'

        return layout

    def create_default_cell_id_layout(self, rows, cols):
        """Create a default cell ID layout"""
        layout = np.empty((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                if self.plate_format == '96-well':
                    if j == cols - 1:
                        layout[i, j] = 'Positive Control'
                    elif j == cols - 2:
                        layout[i, j] = 'Negative Control'
                    else:
                        layout[i, j] = f'ID_{i*cols + j + 1:03d}'
                else:  # 384-well
                    if j >= cols - 4:
                        layout[i, j] = 'Control'
                    else:
                        layout[i, j] = f'ID_{i*cols + j + 1:04d}'

        return layout

    def save_layouts(self):
        """Save current layouts to files"""
        try:
            # Create directory if it doesn't exist
            setup_dir = os.path.join(self.config_manager.config_dir, 'setup_files')
            os.makedirs(setup_dir, exist_ok=True)

            # Save agonist layout
            if self.agonist_layout is not None:
                df = pd.DataFrame(self.agonist_layout)
                df.index = [chr(65 + i) for i in range(self.rows)]
                df.columns = [str(i+1) for i in range(self.cols)]
                self.config_manager.save_plate_layout('agonist_layout.csv', df, overwrite=True)

            # Save cell line layout
            if self.cell_line_layout is not None:
                df = pd.DataFrame(self.cell_line_layout)
                df.index = [chr(65 + i) for i in range(self.rows)]
                df.columns = [str(i+1) for i in range(self.cols)]
                self.config_manager.save_plate_layout('cell_line_layout.csv', df, overwrite=True)

            # Save cell ID layout
            if self.cell_id_layout is not None:
                df = pd.DataFrame(self.cell_id_layout)
                df.index = [chr(65 + i) for i in range(self.rows)]
                df.columns = [str(i+1) for i in range(self.cols)]
                self.config_manager.save_plate_layout('cell_id_layout.csv', df, overwrite=True)

            # New: Save group ID layout
            if self.group_id_layout is not None:
                df = pd.DataFrame(self.group_id_layout)
                df.index = [chr(65 + i) for i in range(self.rows)]
                df.columns = [str(i+1) for i in range(self.cols)]
                self.config_manager.save_plate_layout('group_id_layout.csv', df, overwrite=True)

            QMessageBox.information(self, "Success", "Plate layouts saved successfully.")
            logger.info("Plate layouts saved successfully.")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save layouts: {str(e)}")
            logger.error(f"Error saving plate layouts: {str(e)}", exc_info=True)
            return False

    def load_layouts(self):
        """Load layouts from files"""
        try:
            # Load agonist layout
            agonist_df = self.config_manager.load_plate_layout('agonist_layout.csv')
            if agonist_df is not None:
                self.agonist_layout = np.array(agonist_df)
                self.populate_layout_table(self.agonist_editor, self.agonist_layout)
                self.layout_changed.emit('agonist', self.agonist_layout)

            # Load cell line layout
            cell_line_df = self.config_manager.load_plate_layout('cell_line_layout.csv')
            if cell_line_df is not None:
                self.cell_line_layout = np.array(cell_line_df)
                self.populate_layout_table(self.cell_line_editor, self.cell_line_layout)
                self.layout_changed.emit('cell_line', self.cell_line_layout)

            # Load cell ID layout
            cell_id_df = self.config_manager.load_plate_layout('cell_id_layout.csv')
            if cell_id_df is not None:
                self.cell_id_layout = np.array(cell_id_df)
                # No editor for cell ID layout currently

            # New: Load group ID layout
            group_id_df = self.config_manager.load_plate_layout('group_id_layout.csv')
            if group_id_df is not None:
                self.group_id_layout = np.array(group_id_df)
                self.populate_layout_table(self.group_id_editor, self.group_id_layout)
                self.layout_changed.emit('group_id', self.group_id_layout)

            QMessageBox.information(self, "Success", "Plate layouts loaded successfully.")
            logger.info("Plate layouts loaded successfully.")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load layouts: {str(e)}")
            logger.error(f"Error loading plate layouts: {str(e)}", exc_info=True)
            return False

    def populate_cell_line_list(self):
        """Populate the cell line list from configuration"""
        try:
            # Get cell lines from config
            cell_lines = self.config_manager.config.get('cell_lines', {})

            # Clear current list
            self.cell_line_list.setRowCount(0)

            # Add each cell line to the list
            for cell_line, params in cell_lines.items():
                row = self.cell_line_list.rowCount()
                self.cell_line_list.insertRow(row)

                # Add cell line name
                self.cell_line_list.setItem(row, 0, QTableWidgetItem(cell_line))

                # Add parameters
                if isinstance(params, dict):
                    self.cell_line_list.setItem(row, 1, QTableWidgetItem(str(params.get('baseline', ''))))
                    self.cell_line_list.setItem(row, 2, QTableWidgetItem(str(params.get('peak_ionomycin', ''))))
                    self.cell_line_list.setItem(row, 3, QTableWidgetItem(str(params.get('peak_other', ''))))
                    self.cell_line_list.setItem(row, 4, QTableWidgetItem(str(params.get('rise_rate', ''))))
                    self.cell_line_list.setItem(row, 5, QTableWidgetItem(str(params.get('decay_rate', ''))))

            logger.info(f"Populated cell line list with {self.cell_line_list.rowCount()} items")

        except Exception as e:
            logger.error(f"Error populating cell line list: {str(e)}", exc_info=True)

    def populate_agonist_list(self):
        """Populate the agonist list from configuration"""
        try:
            # Get agonists from config
            agonists = self.config_manager.config.get('agonists', {})

            # Clear current list
            self.agonist_list.setRowCount(0)

            # Add each agonist to the list
            for agonist, factor in agonists.items():
                row = self.agonist_list.rowCount()
                self.agonist_list.insertRow(row)

                # Add agonist name and factor
                self.agonist_list.setItem(row, 0, QTableWidgetItem(agonist))
                self.agonist_list.setItem(row, 1, QTableWidgetItem(str(factor)))

            logger.info(f"Populated agonist list with {self.agonist_list.rowCount()} items")

        except Exception as e:
            logger.error(f"Error populating agonist list: {str(e)}", exc_info=True)

    def add_cell_line(self):
        """Add a new cell line"""
        try:
            # Create dialog for new cell line
            dialog = QDialog(self)
            dialog.setWindowTitle("Add Cell Line")

            # Create form
            form_layout = QFormLayout()

            name_edit = QLineEdit()
            form_layout.addRow("Cell Line Name:", name_edit)

            baseline_edit = QLineEdit("500")
            form_layout.addRow("Baseline:", baseline_edit)

            peak_iono_edit = QLineEdit("4000")
            form_layout.addRow("Peak (Ionomycin):", peak_iono_edit)

            peak_other_edit = QLineEdit("1000")
            form_layout.addRow("Peak (Other):", peak_other_edit)

            rise_rate_edit = QLineEdit("0.1")
            form_layout.addRow("Rise Rate:", rise_rate_edit)

            decay_rate_edit = QLineEdit("0.05")
            form_layout.addRow("Decay Rate:", decay_rate_edit)

            # Add buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Add")
            cancel_button = QPushButton("Cancel")

            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)

            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)

            # Set dialog layout
            main_layout = QVBoxLayout()
            main_layout.addLayout(form_layout)
            main_layout.addLayout(button_layout)

            dialog.setLayout(main_layout)

            # Show dialog
            if dialog.exec_():
                # Get values
                name = name_edit.text().strip()

                if not name:
                    QMessageBox.warning(self, "Warning", "Cell line name cannot be empty.")
                    return

                # Get parameters
                try:
                    baseline = float(baseline_edit.text())
                    peak_ionomycin = float(peak_iono_edit.text())
                    peak_other = float(peak_other_edit.text())
                    rise_rate = float(rise_rate_edit.text())
                    decay_rate = float(decay_rate_edit.text())
                except ValueError:
                    QMessageBox.warning(self, "Warning", "Invalid parameter values. Please enter numeric values.")
                    return

                # Add to config
                cell_lines = self.config_manager.config.get('cell_lines', {})
                cell_lines[name] = {
                    'baseline': baseline,
                    'peak_ionomycin': peak_ionomycin,
                    'peak_other': peak_other,
                    'rise_rate': rise_rate,
                    'decay_rate': decay_rate
                }

                self.config_manager.config['cell_lines'] = cell_lines

                # Update list
                self.populate_cell_line_list()

                logger.info(f"Added new cell line: {name}")

        except Exception as e:
            logger.error(f"Error adding cell line: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add cell line: {str(e)}")

    def edit_cell_line(self):
        """Edit selected cell line"""
        try:
            # Get selected row
            selected_rows = self.cell_line_list.selectedItems()

            if not selected_rows:
                QMessageBox.warning(self, "Warning", "Please select a cell line to edit.")
                return

            row = selected_rows[0].row()
            name_item = self.cell_line_list.item(row, 0)

            if not name_item:
                return

            cell_line_name = name_item.text()
            cell_lines = self.config_manager.config.get('cell_lines', {})

            if cell_line_name not in cell_lines:
                QMessageBox.warning(self, "Warning", f"Cell line '{cell_line_name}' not found in configuration.")
                return

            params = cell_lines[cell_line_name]

            # Create dialog for editing
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Edit Cell Line: {cell_line_name}")

            # Create form
            form_layout = QFormLayout()

            name_edit = QLineEdit(cell_line_name)
            form_layout.addRow("Cell Line Name:", name_edit)

            baseline_edit = QLineEdit(str(params.get('baseline', '')))
            form_layout.addRow("Baseline:", baseline_edit)

            peak_iono_edit = QLineEdit(str(params.get('peak_ionomycin', '')))
            form_layout.addRow("Peak (Ionomycin):", peak_iono_edit)

            peak_other_edit = QLineEdit(str(params.get('peak_other', '')))
            form_layout.addRow("Peak (Other):", peak_other_edit)

            rise_rate_edit = QLineEdit(str(params.get('rise_rate', '')))
            form_layout.addRow("Rise Rate:", rise_rate_edit)

            decay_rate_edit = QLineEdit(str(params.get('decay_rate', '')))
            form_layout.addRow("Decay Rate:", decay_rate_edit)

            # Add buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Update")
            cancel_button = QPushButton("Cancel")

            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)

            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)

            # Set dialog layout
            main_layout = QVBoxLayout()
            main_layout.addLayout(form_layout)
            main_layout.addLayout(button_layout)

            dialog.setLayout(main_layout)

            # Show dialog
            if dialog.exec_():
                # Get values
                new_name = name_edit.text().strip()

                if not new_name:
                    QMessageBox.warning(self, "Warning", "Cell line name cannot be empty.")
                    return

                # Get parameters
                try:
                    baseline = float(baseline_edit.text())
                    peak_ionomycin = float(peak_iono_edit.text())
                    peak_other = float(peak_other_edit.text())
                    rise_rate = float(rise_rate_edit.text())
                    decay_rate = float(decay_rate_edit.text())
                except ValueError:
                    QMessageBox.warning(self, "Warning", "Invalid parameter values. Please enter numeric values.")
                    return

                # Update config
                if new_name != cell_line_name:
                    # Name changed, remove old entry
                    del cell_lines[cell_line_name]

                cell_lines[new_name] = {
                    'baseline': baseline,
                    'peak_ionomycin': peak_ionomycin,
                    'peak_other': peak_other,
                    'rise_rate': rise_rate,
                    'decay_rate': decay_rate
                }

                self.config_manager.config['cell_lines'] = cell_lines

                # Update list
                self.populate_cell_line_list()

                # Update plate layout if needed
                if new_name != cell_line_name and self.cell_line_layout is not None:
                    for i in range(self.rows):
                        for j in range(self.cols):
                            if self.cell_line_layout[i, j] == cell_line_name:
                                self.cell_line_layout[i, j] = new_name

                    # Update editor
                    self.populate_layout_table(self.cell_line_editor, self.cell_line_layout)
                    self.layout_changed.emit('cell_line', self.cell_line_layout)

                logger.info(f"Updated cell line: {cell_line_name} -> {new_name}")

        except Exception as e:
            logger.error(f"Error editing cell line: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to edit cell line: {str(e)}")

    def remove_cell_line(self):
        """Remove selected cell line"""
        try:
            # Get selected row
            selected_rows = self.cell_line_list.selectedItems()

            if not selected_rows:
                QMessageBox.warning(self, "Warning", "Please select a cell line to remove.")
                return

            row = selected_rows[0].row()
            name_item = self.cell_line_list.item(row, 0)

            if not name_item:
                return

            cell_line_name = name_item.text()

            # Confirm deletion
            confirm = QMessageBox.question(
                self, "Confirm Removal",
                f"Are you sure you want to remove the cell line '{cell_line_name}'?\n\n"
                "This will also remove it from the plate layout.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if confirm == QMessageBox.Yes:
                # Remove from config
                cell_lines = self.config_manager.config.get('cell_lines', {})

                if cell_line_name in cell_lines:
                    del cell_lines[cell_line_name]
                    self.config_manager.config['cell_lines'] = cell_lines

                # Update list
                self.populate_cell_line_list()

                # Update plate layout
                if self.cell_line_layout is not None:
                    default_cell = next(iter(cell_lines.keys())) if cell_lines else "Unknown"

                    for i in range(self.rows):
                        for j in range(self.cols):
                            if self.cell_line_layout[i, j] == cell_line_name:
                                self.cell_line_layout[i, j] = default_cell

                    # Update editor
                    self.populate_layout_table(self.cell_line_editor, self.cell_line_layout)
                    self.layout_changed.emit('cell_line', self.cell_line_layout)

                logger.info(f"Removed cell line: {cell_line_name}")

        except Exception as e:
            logger.error(f"Error removing cell line: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to remove cell line: {str(e)}")

    def add_agonist(self):
        """Add a new agonist"""
        try:
            # Create dialog for new agonist
            dialog = QDialog(self)
            dialog.setWindowTitle("Add Agonist")

            # Create form
            form_layout = QFormLayout()

            name_edit = QLineEdit()
            form_layout.addRow("Agonist Name:", name_edit)

            factor_edit = QLineEdit("1.0")
            form_layout.addRow("Response Factor:", factor_edit)

            # Add buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Add")
            cancel_button = QPushButton("Cancel")

            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)

            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)

            # Set dialog layout
            main_layout = QVBoxLayout()
            main_layout.addLayout(form_layout)
            main_layout.addLayout(button_layout)

            dialog.setLayout(main_layout)

            # Show dialog
            if dialog.exec_():
                # Get values
                name = name_edit.text().strip()

                if not name:
                    QMessageBox.warning(self, "Warning", "Agonist name cannot be empty.")
                    return

                # Get factor
                try:
                    factor = float(factor_edit.text())
                except ValueError:
                    QMessageBox.warning(self, "Warning", "Invalid response factor. Please enter a numeric value.")
                    return

                # Add to config
                agonists = self.config_manager.config.get('agonists', {})
                agonists[name] = factor

                self.config_manager.config['agonists'] = agonists

                # Update list
                self.populate_agonist_list()

                logger.info(f"Added new agonist: {name}")

        except Exception as e:
            logger.error(f"Error adding agonist: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add agonist: {str(e)}")

    def edit_agonist(self):
        """Edit selected agonist"""
        try:
            # Get selected row
            selected_rows = self.agonist_list.selectedItems()

            if not selected_rows:
                QMessageBox.warning(self, "Warning", "Please select an agonist to edit.")
                return

            row = selected_rows[0].row()
            name_item = self.agonist_list.item(row, 0)
            factor_item = self.agonist_list.item(row, 1)

            if not name_item or not factor_item:
                return

            agonist_name = name_item.text()
            factor = factor_item.text()

            # Create dialog for editing
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Edit Agonist: {agonist_name}")

            # Create form
            form_layout = QFormLayout()

            name_edit = QLineEdit(agonist_name)
            form_layout.addRow("Agonist Name:", name_edit)

            factor_edit = QLineEdit(factor)
            form_layout.addRow("Response Factor:", factor_edit)

            # Add buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Update")
            cancel_button = QPushButton("Cancel")

            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)

            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)

            # Set dialog layout
            main_layout = QVBoxLayout()
            main_layout.addLayout(form_layout)
            main_layout.addLayout(button_layout)

            dialog.setLayout(main_layout)

            # Show dialog
            if dialog.exec_():
                # Get values
                new_name = name_edit.text().strip()

                if not new_name:
                    QMessageBox.warning(self, "Warning", "Agonist name cannot be empty.")
                    return

                # Get factor
                try:
                    new_factor = float(factor_edit.text())
                except ValueError:
                    QMessageBox.warning(self, "Warning", "Invalid response factor. Please enter a numeric value.")
                    return

                # Update config
                agonists = self.config_manager.config.get('agonists', {})

                if new_name != agonist_name:
                    # Name changed, remove old entry
                    if agonist_name in agonists:
                        del agonists[agonist_name]

                agonists[new_name] = new_factor

                self.config_manager.config['agonists'] = agonists

                # Update list
                self.populate_agonist_list()

                # Update plate layout if needed
                if new_name != agonist_name and self.agonist_layout is not None:
                    for i in range(self.rows):
                        for j in range(self.cols):
                            if self.agonist_layout[i, j] == agonist_name:
                                self.agonist_layout[i, j] = new_name

                    # Update editor
                    self.populate_layout_table(self.agonist_editor, self.agonist_layout)
                    self.layout_changed.emit('agonist', self.agonist_layout)

                logger.info(f"Updated agonist: {agonist_name} -> {new_name}")

        except Exception as e:
            logger.error(f"Error editing agonist: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to edit agonist: {str(e)}")

    def remove_agonist(self):
        """Remove selected agonist"""
        try:
            # Get selected row
            selected_rows = self.agonist_list.selectedItems()

            if not selected_rows:
                QMessageBox.warning(self, "Warning", "Please select an agonist to remove.")
                return

            row = selected_rows[0].row()
            name_item = self.agonist_list.item(row, 0)

            if not name_item:
                return

            agonist_name = name_item.text()

            # Confirm deletion
            confirm = QMessageBox.question(
                self, "Confirm Removal",
                f"Are you sure you want to remove the agonist '{agonist_name}'?\n\n"
                "This will also remove it from the plate layout.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if confirm == QMessageBox.Yes:
                # Remove from config
                agonists = self.config_manager.config.get('agonists', {})

                if agonist_name in agonists:
                    del agonists[agonist_name]
                    self.config_manager.config['agonists'] = agonists

                # Update list
                self.populate_agonist_list()

                # Update plate layout
                if self.agonist_layout is not None:
                    default_agonist = next(iter(agonists.keys())) if agonists else "Unknown"

                    for i in range(self.rows):
                        for j in range(self.cols):
                            if self.agonist_layout[i, j] == agonist_name:
                                self.agonist_layout[i, j] = default_agonist

                    # Update editor
                    self.populate_layout_table(self.agonist_editor, self.agonist_layout)
                    self.layout_changed.emit('agonist', self.agonist_layout)

                logger.info(f"Removed agonist: {agonist_name}")

        except Exception as e:
            logger.error(f"Error removing agonist: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to remove agonist: {str(e)}")

    def apply_pattern(self, layout_type):
        """Apply a pattern to the specified layout"""
        pattern = self.fill_pattern_combo.currentText()

        if layout_type == 'cell_line':
            layout = self.cell_line_layout
            editor = self.cell_line_editor
            items = list(self.config_manager.config.get('cell_lines', {}).keys())
        elif layout_type == 'agonist':
            layout = self.agonist_layout
            editor = self.agonist_editor
            items = list(self.config_manager.config.get('agonists', {}).keys())
        else:
            return

        if not items:
            QMessageBox.warning(self, "Warning", f"No {layout_type}s available to create pattern.")
            return

        # Apply selected pattern
        if pattern == "Checkerboard":
            for i in range(self.rows):
                for j in range(self.cols):
                    idx = (i + j) % len(items)
                    layout[i, j] = items[idx]

        elif pattern == "By Row":
            for i in range(self.rows):
                idx = i % len(items)
                for j in range(self.cols):
                    layout[i, j] = items[idx]

        elif pattern == "By Column":
            for j in range(self.cols):
                idx = j % len(items)
                for i in range(self.rows):
                    layout[i, j] = items[idx]

        elif pattern == "All Same":
            for i in range(self.rows):
                for j in range(self.cols):
                    layout[i, j] = items[0]

        # Update editor and emit change
        self.populate_layout_table(editor, layout)
        self.layout_changed.emit(layout_type, layout)
