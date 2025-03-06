# FLIPR Calcium Response Simulator

A comprehensive Python application for simulating FLIPR (Fluorometric Imaging Plate Reader) calcium signaling data, with support for various cell types, agonists, and error conditions. This simulator is designed to generate realistic calcium response data for testing analysis pipelines, validating diagnostic assays, and exploring error detection algorithms.

![FLIPR Simulator Screenshot](docs/images/application_screenshot.jpg)

## Overview

The FLIPR Calcium Response Simulator provides a realistic simulation of calcium signaling responses based on published data from autism spectrum disorder research. The application generates time-series fluorescence data that mimics experimental FLIPR readings for different cell types (neurotypical, ASD, FXS, etc.) in response to agonists like ATP.

The software offers a graphical user interface that allows users to design plate layouts, configure simulation parameters, introduce various error conditions, and visualize results. Advanced features include batch processing, error comparison, and data export capabilities.

## Features

### Realistic Calcium Response Simulation
- Simulate physiologically realistic calcium responses based on published literature
- Configure multiple cell types with unique response characteristics
- Implement dose-dependent agonist responses with proper EC50 values
- Generate complex multi-well plate experiments with various cell types and treatments
- Adjust noise parameters to create realistic signal variability

### Interactive Plate Layout Design
- Design custom plate layouts with different cell types, agonists, and concentrations
- Visualize layouts with color-coding for easy interpretation
- Save and load plate layouts for reproducible experiments
- Apply patterns to quickly create systematic experimental designs

### Comprehensive Error Simulation
- Simulate 15+ common error types encountered in FLIPR experiments:
  - Cell-based errors (variability, dye loading, health, density)
  - Reagent-based errors (stability, concentration, contamination)
  - Equipment-based errors (camera, liquid handler, timing, focus)
  - Systematic errors (edge effects, temperature, evaporation, crosstalk)
- Control error probability and intensity
- Create custom error types with specific parameters
- Visualize error distribution across the plate
- Run comparison simulations to see error effects

### Advanced Visualization Options
- Multiple visualization modes:
  - All Traces: Overview of all well responses
  - By Cell Line: Grouped/averaged traces by cell type
  - By Agonist: Grouped/averaged traces by agonist
  - Heatmap: Peak response visualization across the plate
  - Single Trace: Detailed view of individual wells
- Interactive well selector grid
- Export publication-quality plots

### Data Export and Analysis
- Export simulation data to CSV or Excel formats
- Configure export settings (file naming, location)
- Auto-save functionality for batch processing
- Comprehensive metadata for experiment tracking

### Debugging and Logging
- Comprehensive logging system
- Interactive debug console
- Save logs for troubleshooting

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
The application requires the following Python packages:
```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
PyQt5>=5.15.0
openpyxl>=3.0.0
```

### Installation Steps
1. Clone the repository:
   ```
   git clone https://github.com/gddickinson/flipr_simulation_advanced.git
   cd flipr-simulator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guide

### Getting Started
Launch the application by running:
```
python main.py
```

### Simulation Tab
The Simulation tab allows you to run simulations and visualize results:

1. **Configure Simulation Parameters**:
   - Set the number of timepoints, time interval, and agonist addition time
   - Adjust noise parameters (read noise, background, photobleaching)
   - Enable/disable random seed for reproducible results

2. **Run Simulation**:
   - Click "Run Simulation" to execute the simulation with current settings
   - Results will be displayed in the visualization area

3. **Visualization Options**:
   - Select different plot types from the dropdown (All Traces, By Cell Line, By Agonist, Heatmap, Single Trace)
   - Use the well selector grid to view individual well responses
   - Export plots for use in presentations or publications

4. **Save/Load/Export**:
   - Save current configuration for future use
   - Load previously saved configurations
   - Export simulation data to CSV or Excel

### Plate Layout Tab
Design your experimental plate layout:

1. **Cell Type Configuration**:
   - Assign cell types to specific columns
   - View and edit cell type properties
   - Apply patterns for quick configuration

2. **Agonist Configuration**:
   - Assign agonists and concentrations to specific rows
   - View and edit agonist properties
   - Adjust concentration values in μM

3. **Plate Visualization**:
   - View color-coded representations of the plate layout
   - Switch between cell type, agonist, and concentration views
   - Save and load layouts for reuse

### Error Simulation Tab
Simulate various error conditions:

1. **Error Type Selection**:
   - Enable specific error types via checkboxes
   - Adjust global error probability and intensity
   - Select from preset error scenarios

2. **Custom Error Configuration**:
   - Design custom error patterns
   - Set specific parameters for each error type
   - Apply to all wells or specific wells
   - Option to use global error settings

3. **Error Visualization**:
   - View heat map showing error distribution across the plate
   - See detailed descriptions of active errors

4. **Error Comparison**:
   - Run comparison simulations (normal vs. with errors)
   - View side-by-side comparison of error effects in a popup window
   - Analyze statistical impact of errors

### Debug Console Tab
Monitor application events and troubleshoot issues:

1. **View Log Messages**:
   - Track simulation progress
   - See error and warning messages
   - Monitor application status

2. **Control Options**:
   - Clear console content
   - Adjust log level
   - Save log to file

### Settings Tab
Configure global application settings:

1. **Output Settings**:
   - Set default output directory
   - Choose default export format (CSV/Excel)
   - Configure auto-save options

2. **File Naming**:
   - Set file naming convention (Timestamp, Incremental, Custom)
   - Configure custom file prefix

3. **Performance Settings**:
   - Adjust number of threads for calculation

## Experimental Design

### Cell Types
The simulator includes several cell types with different calcium response characteristics:

- **Neurotypical**: Standard calcium responses with normal kinetics
- **ASD (Autism Spectrum Disorder)**: Reduced peak responses
- **FXS (Fragile X Syndrome)**: Reduced peaks with slower decay
- **Positive Control**: Enhanced response for validation
- **Negative Control**: Minimal response for baseline comparison

### Agonists
Various agonists with different potencies and EC50 values:

- **ATP**: Standard purinergic agonist (EC50 = 100 μM)
- **UTP**: Alternative purinergic agonist (EC50 = 150 μM)
- **Ionomycin**: Calcium ionophore for maximum response (EC50 = 0.5 μM)
- **Buffer**: Control with minimal response

### Default Plate Layout
The default 96-well plate layout follows a common experimental design:

- **Cell Types (by Column)**: Alternating pattern of cell types for comparison
- **Agonists (by Row)**:
  - Rows A-C: ATP (100 μM)
  - Rows D-F: Ionomycin (1 μM)
  - Rows G-H: Buffer (control)

This provides triplicate measurements for each agonist, duplicate buffer controls, and testing of all cell lines against all agonists.

## Error Models

### Cell-Based Errors
- **Cell Variability**: Introduces increased random variation in cell responses
- **Dye Loading Issues**: Simulates problems with calcium dye loading, causing reduced signal
- **Cell Health Problems**: Unhealthy cells with altered baseline and kinetics
- **Variable Cell Density**: Uneven cell distribution causing signal magnitude differences

### Reagent-Based Errors
- **Reagent Stability Issues**: Degraded reagents with reduced potency
- **Incorrect Concentrations**: Pipetting errors causing concentration variations
- **Reagent Contamination**: Contaminated reagents causing unexpected responses

### Equipment-Based Errors
- **Camera Errors**: Simulates dead pixels, saturation, and signal drops
- **Liquid Handler Issues**: Inaccurate dispensing, timing errors, missed wells
- **Timing Inconsistencies**: Irregular data acquisition timing
- **Focus Problems**: Focus issues affecting signal quality

### Systematic Errors
- **Plate Edge Effects**: Edge wells showing different behavior
- **Temperature Gradients**: Temperature variations across the plate
- **Evaporation**: Signal drift due to sample evaporation
- **Well-to-Well Crosstalk**: Optical crosstalk between adjacent wells

### Custom Errors
- **Random Spikes**: Add random spikes to traces
- **Signal Dropouts**: Periods of reduced signal
- **Baseline Drift**: Rising or falling baseline over time
- **Oscillating Baseline**: Sinusoidal oscillation in baseline
- **Signal Cutout**: Complete signal loss for a period
- **Incomplete Decay**: Signal that doesn't return to baseline
- **Extra Noise**: Additional random noise
- **Overlapping Oscillation**: Sinusoidal component added to signal
- **Sudden Jump**: Abrupt change in signal level
- **Exponential Drift**: Exponentially increasing/decreasing drift
- **Delayed Response**: Delayed response to agonist

## Troubleshooting

### Common Issues
- **Simulation takes a long time**: Reduce the number of timepoints or wells
- **Graph doesn't update**: Switch to a different plot type and back
- **Error during simulation**: Check the Debug Console for detailed error messages

### Error Messages
- **"lam value too large"**: Very large values in simulation; try reducing peak heights
- **"Mean of empty slice"**: Usually harmless warning about empty arrays
- **"No well selected"**: Select a well from the grid when in Single Trace mode

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Based on research by Schmunk et al. on calcium signaling in ASD
- Thanks to the Python scientific computing community
