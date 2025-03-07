# Agonist Response Simulator

A focused Python application for simulating single-well calcium response traces from FLIPR (Fluorometric Imaging Plate Reader) experiments. This simulator generates realistic calcium signaling data in response to various agonists, with customizable signal parameters and error models.

## Overview

The Agonist Response Simulator provides a streamlined interface for generating individual calcium response traces. Unlike the more complex FLIPR Calcium Response Simulator which models entire plate experiments, this application focuses on creating and visualizing single traces with highly customizable parameters.

This tool is particularly useful for:
- Developing and testing analysis algorithms
- Training researchers in calcium signal interpretation
- Exploring the effects of various error conditions on signal quality
- Creating reference datasets with known parameters
- Troubleshooting analysis pipelines

## Features

### Agonist Selection and Dose-Response
- Choose from common agonists (ATP, UTP, Ionomycin, Buffer) or create custom profiles
- Dynamic dose-response curves based on agonist-specific EC50 values
- Adjust concentration to see realistic dose-dependent changes in signal amplitude

### Signal Parameters
- Customize baseline fluorescence level
- Adjust peak height for precise signal-to-baseline ratios
- Control rise and decay kinetics independently
- Realistic noise, background, and photobleaching simulation

### Comprehensive Error Models
- **Random Spikes**: Add random fluorescence spikes with controllable amplitude and frequency
- **Signal Dropouts**: Simulate temporary signal loss with adjustable duration
- **Baseline Drift**: Add linear rising or falling trends
- **Oscillating Baseline**: Create sinusoidal oscillations with adjustable frequency and amplitude
- **Delayed Response**: Shift the response in time to simulate delayed agonist action
- **Incomplete Decay**: Model impaired calcium clearance with signals that don't return to baseline
- **Extra Noise**: Add additional Gaussian noise to decrease signal quality
- **Sudden Jump**: Create abrupt signal level changes
- **Signal Cutout**: Simulate complete signal loss for a defined period
- **Exponential Drift**: Add exponentially increasing or decreasing baseline drift

### Visualization and Analysis
- Real-time visualization of both ideal and noise-affected signals
- Detailed signal information panel showing key parameters
- Export publication-quality plots in multiple formats

### Data Management
- Save and load simulation configurations
- Export data to CSV format with complete parameter metadata
- Comprehensive logging for debugging and documentation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
PyQt5>=5.15.0
```

### Installation Steps
1. Clone the repository or download the source code
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python agonist_response_simulator.py
   ```

## Usage Guide

### Agonist Configuration
1. Select an agonist type from the dropdown menu
2. Adjust the concentration value (in μM)
3. The application automatically sets appropriate signal parameters based on the agonist

### Signal Parameter Adjustment
1. Modify baseline, peak height, rise rate, and decay rate as needed
2. Set acquisition parameters (timepoints, time interval, agonist addition time)
3. Adjust noise parameters (read noise, background, photobleaching)

### Error Simulation
1. Enable "Apply Errors" checkbox to add error models
2. Select the desired error types from the available tabs
3. Configure specific parameters for each selected error type

### Running a Simulation
1. Click "Run Simulation" to generate a trace with current settings
2. View the resulting trace in the visualization panel
3. Check the signal information panel for key metrics

### Exporting Results
1. Use "Export Plot" to save the visualization as an image file
2. Use "Export Data" to save the raw data as a CSV file
3. Configuration parameters are automatically saved with exported data

## Technical Details

### Calcium Response Model
The simulator generates calcium responses based on a mathematical model with parameters calibrated from published data. The model includes:

1. **Baseline Phase**: Constant fluorescence level with noise
2. **Response Phase**: Rapid rise following agonist addition with sigmoidal kinetics
3. **Decay Phase**: Exponential decay back toward baseline

### Dose-Response Relationship
The simulator uses the Hill equation to calculate agonist responses:

Response = [C]^h / (EC50^h + [C]^h)

Where:
- [C] is the concentration
- EC50 is the half-maximal effective concentration
- h is the Hill coefficient (agonist-specific)

### Default Agonist Parameters

| Agonist    | EC50 (μM) | Hill Coefficient |
|------------|-----------|------------------|
| ATP        | 100.0     | 1.5              |
| UTP        | 150.0     | 1.3              |
| Ionomycin  | 0.5       | 2.0              |
| Buffer     | N/A       | 1.0              |
| Custom     | 100.0     | 1.5              |

## Troubleshooting

### Common Issues
- **Simulation runs slowly**: Reduce the number of timepoints
- **No visible response**: Check agonist concentration is above minimal effective dose
- **Export fails**: Ensure you have write permissions for the export directory

### Logging
The application includes a comprehensive logging system:
- View logs in the Debug Console tab
- Adjust log levels (INFO, DEBUG, WARNING, ERROR)
- Save logs to file for detailed troubleshooting

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Based on research by Schmunk et al. on calcium signaling in ASD

