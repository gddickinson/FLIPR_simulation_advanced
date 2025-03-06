# FLIPR Calcium Response Simulator
# User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Main Interface Overview](#main-interface-overview)
4. [Simulation Tab](#simulation-tab)
5. [Plate Layout Tab](#plate-layout-tab)
6. [Error Simulation Tab](#error-simulation-tab)
7. [Batch Processing Tab](#batch-processing-tab)
8. [Debug Console Tab](#debug-console-tab)
9. [Settings Tab](#settings-tab)
10. [Data Export & Import](#data-export--import)
11. [Troubleshooting](#troubleshooting)
12. [Technical Reference](#technical-reference)
13. [Appendix](#appendix)

---

## Introduction

The FLIPR (Fluorometric Imaging Plate Reader) Calcium Response Simulator is a comprehensive software tool for generating realistic calcium signaling data. This application is designed to support research, education, and testing of analysis pipelines for calcium signaling experiments, particularly those focused on autism spectrum disorder (ASD) and related conditions.

### About Calcium Signaling in ASD Research

Calcium signaling plays a critical role in cellular function, particularly in neurons. Research has shown that abnormal calcium signaling is associated with various neurodevelopmental disorders, including ASD, Fragile X Syndrome (FXS), and Tuberous Sclerosis Complex (TSC).

This simulator is based on published data demonstrating that fibroblasts from individuals with these conditions show depressed IP3-mediated calcium release when stimulated with ATP, compared to cells from neurotypical controls. The scientific basis for this simulator comes primarily from the work of Schmunk et al. (Scientific Reports, 2017), which identified calcium signaling dysfunction in both monogenic and sporadic forms of ASD.

### Purpose of the Simulator

This software allows users to:

1. Generate realistic calcium response data for multiple cell types
2. Design and test experimental plate layouts
3. Introduce various types of experimental errors
4. Visualize and analyze the resulting data
5. Batch process multiple simulations
6. Export data for further analysis

The simulator is valuable for:
- Developing and testing analysis algorithms
- Training researchers and students
- Designing experimental protocols
- Exploring the effects of various errors on data quality
- Generating test datasets for diagnostic assays

---

## Installation

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **Processor**: 2 GHz or faster
- **Memory**: 4 GB RAM minimum (8 GB recommended)
- **Disk Space**: 500 MB available
- **Python**: Version 3.8 or higher
- **Display**: 1280 x 800 resolution or higher

### Installation Steps

1. **Install Python**:
   - Download and install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation on Windows

2. **Download the FLIPR Simulator**:
   - Clone the repository or download the ZIP file from GitHub
   ```
   git clone https://github.com/yourusername/flipr-simulator.git
   cd flipr-simulator
   ```

3. **Create a Virtual Environment** (recommended):
   ```
   python -m venv venv
   ```
   - Activate the virtual environment:
     - Windows: `venv\Scripts\activate`
     - macOS/Linux: `source venv/bin/activate`

4. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

5. **Launch the Application**:
   ```
   python main.py
   ```

### Installing Additional Dependencies

If you encounter dependency issues, you may need to install additional packages:

#### Windows

```
pip install pywin32
```

#### macOS

```
pip install pyobjc
```

---

## Main Interface Overview

The FLIPR Simulator interface consists of a main window with several tabs, each dedicated to a specific aspect of the simulation process.

### Window Layout

The main window is divided into the following areas:
- **Menu Bar**: Contains File, Edit, View, and Help menus
- **Tab Bar**: Allows switching between different functional areas
- **Status Bar**: Displays current status and messages

### Tabs Overview

1. **Simulation**: Set up and run simulations, visualize results
2. **Plate Layout**: Design plate layouts with cell types and agonists
3. **Error Simulation**: Configure and apply various error models
4. **Batch Processing**: Set up and run multiple simulations
5. **Debug Console**: View application logs and debugging information
6. **Settings**: Configure application preferences

### General Controls

- Most controls feature tooltips – hover over an item to see a description
- Form fields typically have default values that can be reset
- Changes in one tab may affect options in other tabs

---

## Simulation Tab

The Simulation tab is the primary interface for running individual simulations and visualizing results.

### Simulation Parameters

1. **Plate Type**
   - **96-well**: Standard 8 × 12 microplate format
   - **384-well**: High-throughput 16 × 24 format

2. **Timepoints**
   - **Number of Timepoints**: Total data points to generate (10-2000)
   - **Time Interval (seconds)**: Time between consecutive readings (0.1-10 seconds)
   - **Agonist Addition Time (seconds)**: When ATP or other agonist is added (1-100 seconds)

3. **Noise Parameters**
   - **Read Noise**: Random variation in signal (0-500 units)
   - **Background**: Baseline fluorescence level (0-2000 units)
   - **Photobleaching Rate**: Signal decay due to photobleaching (0-0.05)
   - **Reset to Defaults**: Restore default noise parameters

4. **Random Seed**
   - **Use Random Seed**: When checked, generates different results each time
   - **Random Seed Value**: Set a specific seed for reproducible results

### Running a Simulation

1. Click the "Run Simulation" button to execute the simulation with current settings
2. The status bar will show "Simulation running..." during execution
3. When complete, the visualization area will display the results
4. Note: The plate layout from the Plate Layout tab will be used

### Visualization Options

1. **Plot Type Dropdown**:
   - **All Traces**: Shows all well traces in one plot
   - **By Cell Line**: Groups traces by cell type
   - **By Agonist**: Groups traces by agonist type
   - **Heatmap**: Shows peak responses across the plate
   - **Single Trace**: Shows detailed view of a selected well

2. **Well Selector Grid**:
   - Click on a well button (e.g., A1, B2) to view its individual trace
   - The selected well information appears below the grid

3. **Export Controls**:
   - **Export Plot**: Save the current visualization as an image
   - **Export Data**: Export the simulation data as CSV or Excel

### Configuration Management

- **Save Configuration**: Save current simulation settings to a file
- **Load Configuration**: Load previously saved settings

---

## Plate Layout Tab

The Plate Layout tab allows you to design the experimental layout of the plate, including cell types, agonists, and concentrations.

### Plate Format Settings

- **Plate Format**: Choose between 96-well or 384-well format
- **Fill Pattern**: Select pattern for filling the plate (Checkerboard, By Row, By Column, All Same)

### Cell Line Configuration

1. **Cell Line Assignment**:
   - **Column**: Select a column (1-12)
   - **Cell Type**: Choose a cell type (Neurotypical, ASD, FXS, etc.)
   - **Apply to Column**: Apply the selected cell type to the entire column

2. **Cell Line Properties Table**:
   - View and edit properties for each cell type:
     - **Baseline**: Resting calcium level
     - **Peak (Ionomycin)**: Maximum response to ionomycin
     - **Peak (Other)**: Response to other agonists
     - **Decay Rate**: Calcium signal decay rate

3. **Cell Line Management**:
   - **Add Cell Line**: Create a new cell type
   - **Edit**: Modify a selected cell type
   - **Remove**: Delete a selected cell type

### Agonist Configuration

1. **Agonist Assignment**:
   - **Row**: Select a row (A-H)
   - **Agonist**: Choose an agonist (ATP, UTP, Ionomycin, Buffer)
   - **Concentration**: Set the concentration value
   - **Unit**: Select concentration units (nM, µM, mM)
   - **Apply to Row**: Apply the selected agonist to the entire row

2. **Agonist Properties Table**:
   - View and edit properties for each agonist:
     - **Response Factor**: Relative potency
     - **EC50**: Concentration for half-maximal effect

3. **Agonist Management**:
   - **Add Agonist**: Create a new agonist
   - **Edit**: Modify a selected agonist
   - **Remove**: Delete a selected agonist

### Plate Layout Visualization

The plate layout is displayed in a tabular format with three views:
1. **Cell Lines**: Shows cell type distribution across the plate
2. **Agonists**: Shows agonist distribution across the plate
3. **Concentrations**: Shows concentration values across the plate

### Layout Management

- **Save Layout**: Save the current plate layout to a file
- **Load Layout**: Load a previously saved layout
- **Reset to Default**: Restore the default plate layout

---

## Error Simulation Tab

The Error Simulation tab allows you to introduce various types of errors commonly encountered in FLIPR experiments to test their impact on data quality and analysis algorithms.

### Standard Errors Section

The error types are organized into four categories:

1. **Cell-Based Errors**:
   - **Cell Variability**: Increased variation in cell responses
   - **Dye Loading Issues**: Problems with calcium dye loading
   - **Cell Health Problems**: Unhealthy cells with altered calcium response
   - **Variable Cell Density**: Uneven cell distribution across wells

2. **Reagent-Based Errors**:
   - **Reagent Stability Issues**: Degraded reagents with reduced potency
   - **Incorrect Concentrations**: Pipetting errors causing concentration variations
   - **Reagent Contamination**: Contaminated reagents causing unexpected responses

3. **Equipment-Based Errors**:
   - **Camera Errors**: Camera artifacts and errors
   - **Liquid Handler Issues**: Inaccurate dispensing of reagents
   - **Timing Inconsistencies**: Timing issues with data collection
   - **Focus Problems**: Focus issues affecting signal quality

4. **Systematic Errors**:
   - **Plate Edge Effects**: Edge effects common in microplates
   - **Temperature Gradients**: Temperature variations across the plate
   - **Evaporation**: Evaporation effects over time
   - **Well-to-Well Crosstalk**: Optical crosstalk between adjacent wells

### Custom Errors Section

This section allows you to create highly customizable error types:

1. **Error Type Selector**:
   - **Random Spikes**: Add random spikes to the signal
   - **Signal Dropouts**: Periods of reduced signal
   - **Baseline Drift**: Rising or falling baseline
   - **Oscillating Baseline**: Sinusoidal oscillation in baseline
   - **Signal Cutout**: Complete signal loss for a period
   - **Incomplete Decay**: Signal that doesn't return to baseline
   - **Extra Noise**: Additional random noise
   - **Overlapping Oscillation**: Sinusoidal component added to signal
   - **Sudden Jump**: Abrupt change in signal level
   - **Exponential Drift**: Exponentially increasing/decreasing drift
   - **Delayed Response**: Delayed response to agonist

2. **Error Parameters**:
   - Each error type has specific parameters (amplitude, frequency, etc.)
   - Parameters are dynamically updated based on the selected error type

3. **Well Selection**:
   - **All Wells**: Apply to all wells in the plate
   - **Specific Wells**: Apply to selected wells only (comma-separated list, e.g., "A1,B2,C3")

4. **Global Settings Option**:
   - **Use Global Error Settings**: Apply global error probability and intensity to this custom error

### Global Error Settings

- **Error Probability**: Likelihood of an error affecting each well (0-1)
- **Error Intensity**: Severity of the error effect (0-1)

### Preset Error Scenarios

The dropdown menu provides common error combinations:
- **Custom Settings**: User-defined error settings
- **Dye Loading Issues**: Predominant dye loading problems
- **Cell Health Problems**: Issues with cell health
- **Liquid Handler Failure**: Problems with agonist addition
- **Edge Effects**: Strong edge effects
- **Camera Failure**: Camera artifacts and errors
- **Reagent Degradation**: Degraded reagents with reduced potency
- **Combined Failures**: Multiple errors occurring simultaneously

### Error Visualization

The right panel displays:
- A heatmap showing error distribution across the plate
- A text description of the active errors and their settings

### Control Buttons

- **Apply Error Settings**: Apply the current error settings
- **Run Error Comparison**: Run a comparison between normal and error-affected simulations
- **Clear All Errors**: Reset all error settings

### Error Comparison Window

When running an error comparison, a popup window shows:
- Individual well comparison between normal and error-affected traces
- Statistical comparison of peak responses
- Overview of all traces with and without errors

---

## Batch Processing Tab

The Batch Processing tab allows you to set up and run multiple simulations with different configurations, useful for generating datasets for algorithm testing or exploring parameter spaces.

### Batch Configuration

1. **Batch List**:
   - Displays all configurations in the current batch
   - Each entry shows the configuration name and active error types

2. **List Controls**:
   - **Add Configuration**: Add the current simulation settings to the batch
   - **Duplicate**: Create a copy of the selected configuration
   - **Edit**: Modify the selected configuration
   - **Remove**: Delete the selected configuration from the batch

3. **Preset Scenarios**:
   - **Add Preset Error Scenario**: Select a preset error scenario
   - **Add to Batch**: Add the selected preset to the batch list

### Execution Settings

1. **Output Directory**:
   - Specify where batch results will be saved
   - Use the "Browse..." button to select a directory

2. **Iterations per Configuration**:
   - Number of times to run each configuration (1-100)
   - Useful for generating statistical data or using different random seeds

### Batch Progress

1. **Progress Display**:
   - Shows current batch processing status
   - Displays number of completed simulations

2. **Results Table**:
   - Shows index, configuration, status, and result path for each simulation
   - Updates in real-time as simulations complete

### Control Buttons

- **Run Batch**: Start processing the batch
- **Stop**: Halt the batch processing
- **Export Results**: Export the batch results to a file

---

## Debug Console Tab

The Debug Console tab provides detailed information about application operations, useful for troubleshooting and monitoring simulation progress.

### Console Display

- Displays timestamped log messages
- Color-coded by severity (info, warning, error)
- Automatically scrolls to show newest messages

### Console Controls

1. **Log Level**:
   - **INFO**: Standard operational messages
   - **DEBUG**: Detailed information for debugging
   - **WARNING**: Potential issues that don't prevent operation
   - **ERROR**: Serious problems that may affect functionality

2. **Control Buttons**:
   - **Clear Console**: Remove all messages from the display
   - **Save Log**: Save the current log to a text file

### Interpreting Messages

- **INFO messages** (white): Normal operations
- **DEBUG messages** (gray): Detailed technical information
- **WARNING messages** (orange): Potential issues to be aware of
- **ERROR messages** (red): Problems requiring attention

### Common Messages

- "Simulation completed successfully": Normal completion
- "Error in simulation thread": Problem during simulation
- "Configuration saved/loaded": File operations completed
- "Layout saved/loaded": Plate layout file operations

---

## Settings Tab

The Settings tab allows you to configure global application preferences and default values.

### Output Settings

1. **Output Directory**:
   - Specify the default location for saving results
   - Use "Browse..." to select a directory

2. **Default Output Format**:
   - **CSV**: Comma-separated values format (best for compatibility)
   - **Excel**: Microsoft Excel format (better for visualization)

3. **Auto-save Results**:
   - When checked, automatically saves results after each simulation
   - Uses the settings below for file naming

### File Naming Options

1. **File Naming Convention**:
   - **Timestamp**: Uses date and time (e.g., FLIPR_20230601_143022)
   - **Incremental Number**: Uses sequential numbering (e.g., FLIPR_001)
   - **Custom Prefix**: Uses only the specified prefix

2. **Custom Prefix**:
   - Base name for saved files (default: "FLIPR_simulation")

### Advanced Settings

1. **Number of Threads**:
   - How many CPU threads to use for calculation (1-16)
   - Higher values may improve performance on multi-core systems

### Control Buttons

- **Save Settings**: Save current settings as defaults for future sessions

---

## Data Export & Import

### Exporting Simulation Data

1. **From the Simulation Tab**:
   - Click the "Export Data" button after running a simulation
   - Choose the format (CSV or Excel) based on settings
   - Data will be saved to the configured output directory

2. **Automatic Export**:
   - If "Auto-save Results" is enabled in Settings
   - Occurs automatically after each simulation
   - Uses file naming convention from Settings

### Export File Structure

1. **CSV Export**:
   - **traces.csv**: Time series data for each well
   - **metadata.csv**: Well information (cell type, agonist, etc.)
   - **parameters.json**: Simulation parameters

2. **Excel Export**:
   - **Traces** sheet: Time series data for each well
   - **Metadata** sheet: Well information
   - **Parameters** sheet: Simulation parameters

### Exporting Plots

1. From the Simulation tab, click "Export Plot"
2. Choose the file format (PNG, PDF, SVG)
3. Select the save location
4. The current visualization will be saved as an image

### Importing Configurations

1. From the Simulation tab, click "Load Configuration"
2. Browse to the saved configuration file (.json)
3. Select the file to load the settings

### Importing Plate Layouts

1. From the Plate Layout tab, click "Load Layout"
2. Browse to the saved layout file (.json)
3. Select the file to load the plate layout

---

## Troubleshooting

### Common Issues and Solutions

1. **Application fails to start**:
   - Ensure Python 3.8+ is installed and in your PATH
   - Check that all dependencies are installed (`pip install -r requirements.txt`)
   - Try running from the command line to see error messages

2. **Simulation runs slowly**:
   - Reduce the number of timepoints
   - Use fewer wells or a smaller plate format
   - Close other applications to free up memory
   - Increase the number of threads in Settings

3. **Graph doesn't update**:
   - Switch to a different plot type and back
   - Run the simulation again
   - Restart the application if persistent

4. **Export fails**:
   - Check write permissions for the output directory
   - Ensure the file is not open in another application
   - Use a different output directory

### Error Messages and Meaning

1. **"lam value too large"**:
   - Very large values in simulation
   - Try reducing peak heights or using different cell parameters

2. **"Mean of empty slice"**:
   - Usually harmless warning about empty arrays
   - Can occur with certain plot types and empty wells

3. **"No well selected"**:
   - Appears when using Single Trace mode without selecting a well
   - Select a well from the grid to resolve

4. **"Error in simulation thread"**:
   - General simulation error
   - Check the Debug Console for detailed information

### Getting Help

If you encounter issues not covered here:
1. Check the Debug Console for detailed error messages
2. Consult the FLIPR Simulator GitHub repository for known issues
3. Submit a bug report with the error message and steps to reproduce

---

## Technical Reference

### Calcium Response Model

The simulator generates calcium responses based on a mathematical model with parameters calibrated from published data. The model includes:

1. **Baseline Phase**:
   - Constant fluorescence level with noise
   - Affected by photobleaching over time

2. **Response Phase**:
   - Rapid rise following agonist addition
   - Peak amplitude depends on cell type and agonist
   - Rise kinetics controlled by rise rate parameter

3. **Decay Phase**:
   - Exponential decay back toward baseline
   - Decay rate depends on cell type
   - May not return fully to baseline

### Cell Type Parameters

| Cell Type     | Baseline | Peak (Ionomycin) | Peak (Other) | Rise Rate | Decay Rate |
|---------------|----------|------------------|--------------|-----------|------------|
| Neurotypical  | 500      | 3900             | 750          | 0.05      | 0.05       |
| ASD           | 500      | 3800             | 500          | 0.10      | 0.03       |
| FXS           | 500      | 3700             | 400          | 0.07      | 0.02       |
| Pos. Control  | 500      | 4000             | 1000         | 0.10      | 0.06       |
| Neg. Control  | 500      | 3800             | 325          | 0.10      | 0.01       |

### Agonist Parameters

| Agonist    | Response Factor | EC50 (μM) |
|------------|----------------|-----------|
| ATP        | 2.0            | 100.0     |
| UTP        | 1.8            | 150.0     |
| Ionomycin  | 3.0            | 0.5       |
| Buffer     | 0.1            | N/A       |

### Dose-Response Relationship

The simulator uses the Hill equation to calculate agonist responses:

Response = [C]^h / (EC50^h + [C]^h)

Where:
- [C] is the concentration
- EC50 is the half-maximal effective concentration
- h is the Hill coefficient (typically 1.0-2.0)

### Error Model Implementation

Each error type modifies the calcium signal in specific ways:

1. **Cell Variability**:
   - Randomly scales peak height and kinetics
   - Higher intensity increases variation

2. **Dye Loading Issues**:
   - Scales overall signal intensity
   - Higher intensity decreases signal

3. **Liquid Handler**:
   - Can cause missed additions, wrong timing, or double additions
   - Probability determines likelihood of occurrence

See the code documentation for detailed implementation of each error type.

---

## Appendix

### Default Plate Layout

The default 96-well plate layout follows a common experimental design:

- **Cell Types (by Column)**:
  - Columns 1, 4, 5, 10, 11: ASD
  - Columns 2, 7: FXS
  - Columns 3, 6, 8, 9, 12: Neurotypical

- **Agonists (by Row)**:
  - Rows A-C: ATP (100 μM)
  - Rows D-F: Ionomycin (1 μM)
  - Rows G-H: Buffer (control)

This provides triplicate measurements for each agonist, duplicate buffer controls, and testing of all cell lines against all agonists.

### Keyboard Shortcuts

| Function             | Shortcut      |
|----------------------|---------------|
| Run Simulation       | Ctrl+R        |
| Export Data          | Ctrl+E        |
| Save Configuration   | Ctrl+S        |
| Load Configuration   | Ctrl+O        |
| Exit Application     | Alt+F4        |
| Switch to Next Tab   | Ctrl+Tab      |
| Switch to Previous Tab | Ctrl+Shift+Tab |
| Clear Console        | Ctrl+L        |
| Help                 | F1            |

### File Formats

1. **Configuration Files** (.json):
   - Store simulation parameters
   - Can be shared between users

2. **Plate Layout Files** (.json):
   - Store cell type and agonist arrangements
   - Compatible between app versions

3. **Export Formats**:
   - CSV (.csv): Best for data analysis
   - Excel (.xlsx): Best for viewing and simple analysis
   - Image (.png, .pdf, .svg): For presentations and documentation

### Additional Resources

- **Original Research Paper**: Schmunk et al. (2017). High-throughput screen detects calcium signaling dysfunction in typical sporadic autism spectrum disorder. Scientific Reports, 7, 40740.
- **GitHub Repository**: [github.com/gddickinson/flipr_simulation_advanced](https://github.com/gddickinson/flipr_simulation_advanced)
- **Issue Tracker**: For bug reports and feature requests
