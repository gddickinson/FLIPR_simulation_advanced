import numpy as np
import pandas as pd
import os
import logging
import random
from scipy.stats import norm
from datetime import datetime
from scipy.interpolate import interp1d

logger = logging.getLogger('FLIPR_Simulator.SimulationEngine')

class SimulationEngine:
    """Core engine for simulating FLIPR calcium responses and various error conditions"""

    def __init__(self):
        """Initialize the simulation engine with default parameters"""
        # Default cell line characteristics - can be overridden by config
        self.cell_lines = {
            'Neurotypical': {
                'baseline': 500,
                'peak_ionomycin': 3900,
                'peak_other': 750,
                'rise_rate_ionomycin': 0.07,
                'rise_rate_other': 0.05,
                'decay_rate_ionomycin': 0.06,
                'decay_rate_other': 0.05
            },
            'ASD': {
                'baseline': 500,
                'peak_ionomycin': 3800,
                'peak_other': 500,
                'rise_rate_ionomycin': 0.12,
                'rise_rate_other': 0.10,
                'decay_rate_ionomycin': 0.04,
                'decay_rate_other': 0.03
            },
            'FXS': {
                'baseline': 500,
                'peak_ionomycin': 3700,
                'peak_other': 400,
                'rise_rate_ionomycin': 0.09,
                'rise_rate_other': 0.07,
                'decay_rate_ionomycin': 0.03,
                'decay_rate_other': 0.02
            },
            'NTC': {
                'baseline': 100,
                'peak_ionomycin': 110,
                'peak_other': 105,
                'rise_rate_ionomycin': 0.01,
                'rise_rate_other': 0.01,
                'decay_rate_ionomycin': 0.01,
                'decay_rate_other': 0.01
            },
        }

        # Default agonist characteristics
        self.agonists = {
            'ATP': 2.0,
            'UTP': 1.8,
            'Ionomycin': 3.0,
            'Buffer': 0.1,  # No effect on peak response
        }


        # Set up default error models
        self.error_models = {
            # Cell-based errors
            'cell_variability': self._apply_cell_variability,
            'dye_loading': self._apply_dye_loading_issues,
            'cell_health': self._apply_cell_health_problems,
            'cell_density': self._apply_variable_cell_density,

            # Reagent-based errors
            'reagent_stability': self._apply_reagent_stability_issues,
            'reagent_concentration': self._apply_incorrect_concentrations,
            'reagent_contamination': self._apply_reagent_contamination,

            # Equipment-based errors
            'camera_errors': self._apply_camera_errors,
            'liquid_handler': self._apply_liquid_handler_issues,
            'timing_errors': self._apply_timing_inconsistencies,
            'focus_problems': self._apply_focus_problems,

            # Systematic errors
            'edge_effects': self._apply_edge_effects,
            'temperature_gradient': self._apply_temperature_gradient,
            'evaporation': self._apply_evaporation,
            'well_crosstalk': self._apply_well_crosstalk,

            # Add the custom error model
            'custom_error': self._apply_custom_error

        }

        # Default simulation parameters
        self.default_params = {
            'num_wells': 96,
            'num_timepoints': 451,
            'time_interval': 0.4,  # seconds
            'agonist_addition_time': 10,  # seconds
            'read_noise': 20,
            'background': 100,
            'photobleaching_rate': 0.0005,
            'random_seed': 42
        }



    def simulate(self, config):
        """
        Run a full plate simulation with the given configuration

        Args:
            config (dict): Configuration parameters for the simulation

        Returns:
            dict: Simulation results including plate data and metadata
        """
        # Merge config with defaults
        params = {**self.default_params, **config}

        # Set random seed if specified
        if 'random_seed' in params and params['random_seed'] is not None:
            np.random.seed(params['random_seed'])
            random.seed(params['random_seed'])

        # Ensure cell line and agonist properties from config are used
        if 'cell_lines' in params:
            for cell_line, properties in params['cell_lines'].items():
                if cell_line in self.cell_lines:
                    # Update existing cell line properties
                    self.cell_lines[cell_line].update(properties)
                else:
                    # Add new cell line
                    self.cell_lines[cell_line] = properties

        if 'agonists' in params:
            self.agonists.update(params['agonists'])

        logger.info(f"Starting simulation with {params['num_timepoints']} timepoints")

        # Calculate total time
        total_time = params['num_timepoints'] * params['time_interval']

        # Get plate dimensions based on plate type
        if params.get('plate_type', '96-well') == '96-well':
            rows, cols = 8, 12
        else:
            rows, cols = 16, 24

        # Get plate layouts from config or create defaults
        if 'cell_line_layout' in params:
            cell_line_layout = np.array(params['cell_line_layout'])
        else:
            cell_line_layout = self._create_default_cell_line_layout(rows, cols, 'Neurotypical')

        if 'agonist_layout' in params:
            agonist_layout = np.array(params['agonist_layout'])
        else:
            agonist_layout = self._create_default_agonist_layout(rows, cols, 'ATP')

        # Get group ID layout or create default
        if 'group_id_layout' in params:
            group_id_layout = np.array(params['group_id_layout'])
        else:
            # Create a default group layout if none provided
            group_id_layout = np.empty((rows, cols), dtype=object)
            for j in range(cols):
                group_id = f"Group {chr(65 + (j % 4))}"  # Group A, B, C, D pattern
                for i in range(rows):
                    group_id_layout[i, j] = group_id

        # Create default cell ID layout
        cell_id_layout = self._create_default_cell_id_layout(rows * cols)

        # Get concentration layout or create default
        if 'concentration_layout' in params:
            concentration_layout = np.array(params['concentration_layout'], dtype=float)
        else:
            concentration_layout = np.ones((rows, cols)) * 100.0  # Default 100µM everywhere

        # Initialize plate data array and metadata
        plate_data = np.zeros((rows * cols, params['num_timepoints']))
        metadata = []

        # Generate responses for each well
        for row in range(rows):
            for col in range(cols):
                well = row * cols + col

                cell_line = cell_line_layout[row, col]
                agonist = agonist_layout[row, col]
                cell_id = f"{chr(65 + row)}{col + 1}"  # e.g., A1, B5, etc.
                concentration = concentration_layout[row, col]
                # Get group ID for this well
                group_id = group_id_layout[row, col]

                if cell_line not in self.cell_lines or agonist not in self.agonists:
                    # Use randomization settings if available
                    randomization = params.get('randomization')

                    plate_data[well] = self._add_realistic_noise(
                        np.zeros(params['num_timepoints']),
                        params['read_noise'],
                        params['background'],
                        params['photobleaching_rate'],
                        randomization
                    )

                    metadata.append({
                        'well_id': cell_id,
                        'cell_id': cell_id,
                        'cell_line': cell_line,
                        'agonist': agonist,
                        'concentration': concentration,
                        'group_id': group_id,
                        'valid': False,
                        'error': 'Invalid cell line or agonist'
                    })
                    continue

                # Get normal response
                cell_params = self.cell_lines[cell_line]

                # Calculate dose-response factor
                dose_response_factor = self._calculate_dose_response(agonist, concentration)

                # Get agonist's intrinsic potency factor
                agonist_factor = self.agonists[agonist]

                # Calculate final response factor
                response_factor = agonist_factor * dose_response_factor

                # Generate calcium response with randomization settings if available
                randomization = params.get('randomization')

                response = self._generate_calcium_response(
                    cell_params['baseline'],
                    cell_params['peak_ionomycin'],
                    cell_params['peak_other'],
                    cell_params.get('rise_rate_ionomycin', cell_params.get('rise_rate', 0.1)),
                    cell_params.get('rise_rate_other', cell_params.get('rise_rate', 0.07)),
                    cell_params.get('decay_rate_ionomycin', cell_params.get('decay_rate', 0.05)),
                    cell_params.get('decay_rate_other', cell_params.get('decay_rate', 0.03)),
                    response_factor,
                    agonist,
                    params['time_interval'],
                    total_time,
                    params['agonist_addition_time'],
                    params['num_timepoints'],
                    randomization
                )

                # Apply any active error models
                if 'active_errors' in params:
                    for error_type, settings in params['active_errors'].items():
                        if error_type in self.error_models and random.random() < settings.get('probability', 0):
                            response = self.error_models[error_type](response, well, row, col, params, settings)

                # Add noise to the response
                noisy_response = self._add_realistic_noise(
                    response,
                    params['read_noise'],
                    params['background'],
                    params['photobleaching_rate'],
                    randomization
                )

                # Apply cell-to-cell variability if enabled
                if randomization and 'cell_variability' in randomization:
                    cell_var_settings = randomization['cell_variability']
                    if cell_var_settings.get('enabled', True):
                        # Find non-baseline portion of the response
                        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

                        # Calculate the peak of the response
                        if len(noisy_response) > baseline_end:
                            baseline = np.mean(noisy_response[:baseline_end]) if baseline_end > 0 else noisy_response[0]

                            # Apply random scaling to the peak response part
                            variability = cell_var_settings.get('amount', 0.2)
                            scale_factor = 1.0 + np.random.normal(0, variability)
                            scale_factor = max(0.1, scale_factor)  # Ensure it doesn't go too low

                            # Apply the scaling while preserving baseline
                            noisy_response[baseline_end:] = baseline + scale_factor * (noisy_response[baseline_end:] - baseline)

                plate_data[well] = noisy_response

                # Add metadata for this well
                metadata.append({
                    'well_id': cell_id,
                    'cell_id': cell_id,
                    'cell_line': cell_line,
                    'agonist': agonist,
                    'concentration': concentration,
                    'response_factor': response_factor,
                    'group_id': group_id,
                    'valid': True
                })

        logger.info(f"Simulation completed for {rows * cols} wells")

        # Initialize results dictionary
        results = {
            'plate_data': plate_data,
            'metadata': metadata,
            'params': params,
            'time_points': np.arange(0, total_time, params['time_interval']),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Calculate DF/F0 if requested
        simulate_df_f0 = params.get('simulate_df_f0', False)
        if simulate_df_f0:
            # Calculate F0 for each well
            df_f0_data = np.zeros_like(plate_data)
            baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

            for well in range(len(plate_data)):
                # Only process if we have enough baseline points
                if baseline_end > 0 and baseline_end < len(plate_data[well]):
                    # Calculate F0 as mean of baseline period
                    f0 = np.mean(plate_data[well][:baseline_end])

                    # Avoid division by zero
                    if f0 > 0:
                        # Calculate DF/F0
                        if params.get('df_f0_as_percent', True):
                            df_f0_data[well] = ((plate_data[well] - f0) / f0) * 100  # Express as percentage
                        else:
                            df_f0_data[well] = (plate_data[well] - f0) / f0  # Express as ratio
                    else:
                        # If baseline is zero or negative, just set to zeros
                        df_f0_data[well] = np.zeros_like(plate_data[well])
                        logger.warning(f"Zero or negative baseline detected in well {well}, unable to calculate DF/F0")

            # Add DF/F0 data to results
            results['df_f0_data'] = df_f0_data

        #DEBUG
        if simulate_df_f0:
            # After calculating df_f0_data
            logger.info(f"DF/F0 calculation enabled: {'percentage' if params.get('df_f0_as_percent', True) else 'ratio'} mode")
            logger.info(f"DF/F0 data shape: {df_f0_data.shape}, min: {np.min(df_f0_data)}, max: {np.max(df_f0_data)}")

            # Debug: Check a sample well
            sample_well = 0
            baseline_end = int(params['agonist_addition_time'] / params['time_interval'])
            sample_f0 = np.mean(plate_data[sample_well][:baseline_end])
            logger.info(f"Sample well F0: {sample_f0}, min: {np.min(plate_data[sample_well])}, max: {np.max(plate_data[sample_well])}")
            logger.info(f"Sample well DF/F0 min: {np.min(df_f0_data[sample_well])}, max: {np.max(df_f0_data[sample_well])}")

        return results

    def _create_default_agonist_layout(self, rows, cols, default_agonist='ATP'):
        """Create a default agonist layout with the specified default agonist"""
        layout = np.empty((rows, cols), dtype=object)

        # Fill most wells with the default agonist
        layout.fill(default_agonist)

        # Always include some controls
        # Add Ionomycin in rightmost columns as positive control
        for i in range(rows):
            layout[i, cols-1] = 'Ionomycin'
            layout[i, cols-2] = 'Ionomycin'

        # Add Buffer in some wells as negative control
        for i in range(rows):
            if i % 2 == 0:  # Even rows
                layout[i, cols-3] = 'Buffer'

        return layout

    def _create_default_cell_line_layout(self, rows, cols, default_cell_line='Neurotypical'):
        """Create a default cell line layout with the specified default cell line"""
        layout = np.empty((rows, cols), dtype=object)

        # Fill with default cell line first
        layout.fill(default_cell_line)

        # EXPLICITLY assign by column index to avoid confusion
        for i in range(rows):
            # Column 12 (index 11) - Neurotypical
            layout[i, 11] = 'Neurotypical'

            # Column 11 (index 10) - NTC
            layout[i, 10] = 'NTC'

            # Column 10 (index 9) - ASD
            layout[i, 9] = 'ASD'

            # Column 9 (index 8) - Some FXS samples
            if i % 2 == 0:  # Even rows
                layout[i, 8] = 'FXS'

        return layout

    def _add_random_amount(self, value, amount=200, enable=True):
        """Add a random amount to a value within the specified range"""
        if not enable:
            return value

        random_amount = random.randint(-amount, amount)
        new_value = value + random_amount
        return max(0, new_value)  # Ensure value is non-negative

    def _generate_calcium_response(self, baseline, peak_ionomycin, peak_other,
                                 rise_rate_ionomycin, rise_rate_other,
                                 decay_rate_ionomycin, decay_rate_other,
                                 agonist_factor, agonist, time_interval,
                                 total_time, agonist_addition_time, num_timepoints,
                                 randomization=None):
        """Generate a calcium response curve based on the parameters"""
        # Create time array
        time = np.arange(0, total_time, time_interval)
        response = np.zeros(num_timepoints)

        # Apply randomization if configured
        if randomization is None:
            randomization = {
                'baseline': {'enabled': True, 'amount': 200},
                'peak': {'enabled': True, 'amount': 200}
            }

        # Add some randomness to baseline if enabled
        baseline_settings = randomization.get('baseline', {'enabled': True, 'amount': 200})
        baseline = self._add_random_amount(baseline,
                                         amount=baseline_settings.get('amount', 200),
                                         enable=baseline_settings.get('enabled', True))

        # Set baseline values
        response[:int(agonist_addition_time / time_interval)] = baseline

        # Determine peak response time and value
        peak_index = int((agonist_addition_time + 5) / time_interval)  # Peak 5 seconds after agonist addition
        peak_time = time[peak_index]

        # Select appropriate peak, rise rate, and decay rate based on agonist
        if agonist == 'Ionomycin':
            peak = peak_ionomycin
            rise_rate = rise_rate_ionomycin
            decay_rate = decay_rate_ionomycin
        else:
            peak = peak_other
            rise_rate = rise_rate_other
            decay_rate = decay_rate_other

        # Add randomness to peak if enabled
        peak_settings = randomization.get('peak', {'enabled': True, 'amount': 200})
        peak = self._add_random_amount(peak,
                                     amount=peak_settings.get('amount', 200),
                                     enable=peak_settings.get('enabled', True))

        # Calculate adjusted peak based on agonist factor
        adjusted_peak = baseline + (peak - baseline) * agonist_factor

        # Generate response curve with rise and decay
        response[int(agonist_addition_time / time_interval):] = baseline + (adjusted_peak - baseline) * (
            norm.pdf(time[int(agonist_addition_time / time_interval):], peak_time, 1/rise_rate) /
            norm.pdf(peak_time, peak_time, 1/rise_rate)
        ) * np.exp(-decay_rate * (time[int(agonist_addition_time / time_interval):] - peak_time))

        return response

    # In SimulationEngine class
    def _add_realistic_noise(self, signal, read_noise=20, background=100, photobleaching_rate=0.0005, randomization=None):
        """Add realistic noise to the fluorescence signal"""
        # Ensure signal is float
        signal = signal.astype(float)

        # Use defaults if randomization is not provided
        if randomization is None:
            randomization = {
                'shot_noise': True,
                'read_noise': {'enabled': True, 'amount': read_noise},
                'background': {'enabled': True, 'amount': background},
                'photobleaching': {'enabled': True, 'rate': photobleaching_rate}
            }

        # Ensure all values are in a valid range for Poisson noise
        # NumPy's Poisson has issues with very small or very large lambda values
        # For large values, we'll use Gaussian approximation of Poisson

        # First handle negative or very small values
        signal = np.maximum(signal, 0.01)  # Set a small positive minimum

        # Add shot noise (Poisson or Gaussian approximation) if enabled
        if randomization.get('shot_noise', True):
            noisy_signal = np.empty_like(signal)

            # Use threshold to determine which method to use
            # NumPy's poisson implementation typically has issues when lambda > 10^7
            poisson_threshold = 1e7

            # Split the signal into manageable and large values
            small_mask = signal < poisson_threshold
            large_mask = ~small_mask

            # For small values, use actual Poisson distribution
            if np.any(small_mask):
                noisy_signal[small_mask] = np.random.poisson(signal[small_mask]).astype(float)

            # For large values, use Gaussian approximation of Poisson: N(λ, λ)
            # For Poisson distribution with large lambda, it approaches N(λ, λ)
            if np.any(large_mask):
                large_values = signal[large_mask]
                std_dev = np.sqrt(large_values)  # Standard deviation is sqrt(lambda) for Poisson
                noisy_signal[large_mask] = np.random.normal(large_values, std_dev)
        else:
            # If shot noise is disabled, just copy the original signal
            noisy_signal = signal.copy()

        # Read noise (Gaussian) if enabled
        read_noise_settings = randomization.get('read_noise', {'enabled': True, 'amount': read_noise})
        if read_noise_settings.get('enabled', True):
            noise_amount = read_noise_settings.get('amount', read_noise)
            noisy_signal += np.random.normal(0, noise_amount, signal.shape)

        # Background noise if enabled
        bg_settings = randomization.get('background', {'enabled': True, 'amount': background})
        if bg_settings.get('enabled', True):
            bg_amount = bg_settings.get('amount', background)
            noisy_signal += bg_amount

        # Photobleaching if enabled
        photobleach_settings = randomization.get('photobleaching', {'enabled': True, 'rate': photobleaching_rate})
        if photobleach_settings.get('enabled', True):
            bleach_rate = photobleach_settings.get('rate', photobleaching_rate)
            time = np.arange(len(signal))
            photobleaching = np.exp(-bleach_rate * time)
            noisy_signal *= photobleaching

        return np.maximum(noisy_signal, 0).round(2)  # Ensure non-negative values and round


    # Default layout generators
    def _create_default_cell_id_layout(self, num_wells):
        """Create a default cell ID layout for the plate"""
        rows, cols = 8, num_wells // 8
        layout = np.empty((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                if j == cols - 1:
                    layout[i, j] = 'Positive Control'
                elif j == cols - 2:
                    layout[i, j] = 'Negative Control'
                else:
                    layout[i, j] = f'ID_{i*cols + j + 1:03d}'

        return layout

    # Error simulation methods
    def _apply_cell_variability(self, response, well, row, col, params, settings):
        """Apply increased variability in cell responses"""
        variability = settings.get('intensity', 0.5)
        # Increase the random variation in peak height and kinetics
        perturbed_response = response.copy()

        # Find non-baseline portion of the response
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

        # Calculate the peak of the response
        if len(response) > baseline_end:
            peak_index = baseline_end + np.argmax(response[baseline_end:])
            peak_value = response[peak_index]
            baseline = np.mean(response[:baseline_end])

            # Apply random scaling to the peak
            scale_factor = 1.0 + np.random.normal(0, variability)
            scale_factor = max(0.1, scale_factor)  # Ensure it doesn't go too low

            # Apply the scaling
            perturbed_response[baseline_end:] = baseline + scale_factor * (response[baseline_end:] - baseline)

        return perturbed_response

    def _apply_dye_loading_issues(self, response, well, row, col, params, settings):
        """Simulate problems with dye loading in cells"""
        intensity = settings.get('intensity', 0.5)

        # Dye loading issues mainly affect the baseline and overall signal intensity
        baseline_scaling = max(0.2, 1.0 - intensity * np.random.random())

        # Scale the entire response, simulating reduced dye loading
        return response * baseline_scaling

    def _apply_cell_health_problems(self, response, well, row, col, params, settings):
        """Simulate unhealthy cells with altered calcium responses"""
        intensity = settings.get('intensity', 0.5)

        # Find baseline and post-stimulation portions
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

        if len(response) <= baseline_end:
            return response

        # Get baseline and peak
        baseline = np.mean(response[:baseline_end])
        peak_index = baseline_end + np.argmax(response[baseline_end:])

        # Unhealthy cells may have:
        # 1. Higher baseline calcium (stressed cells)
        # 2. Lower peak response
        # 3. Slower decay rate

        # Adjust baseline (increase)
        baseline_shift = 1.0 + intensity * np.random.random()
        altered_response = response.copy()
        altered_response[:baseline_end] = response[:baseline_end] * baseline_shift

        # Reduce peak height
        peak_reduction = max(0.2, 1.0 - intensity * np.random.random())
        post_stim = altered_response[baseline_end:]
        post_stim = baseline + (post_stim - baseline) * peak_reduction

        # Slower decay - flatten the tail of the response
        if len(post_stim) > 10:
            decay_start = np.argmax(post_stim)
            if decay_start < len(post_stim) - 1:
                decay_portion = post_stim[decay_start:]
                # Make decay more gradual
                decay_factor = 0.7 - 0.4 * intensity * np.random.random()  # Between 0.3-0.7
                new_decay = baseline + (decay_portion[0] - baseline) * np.exp(
                    -decay_factor * np.arange(len(decay_portion)) / len(decay_portion)
                )
                post_stim[decay_start:] = new_decay

        altered_response[baseline_end:] = post_stim
        return altered_response

    def _apply_variable_cell_density(self, response, well, row, col, params, settings):
        """Simulate variable cell density across wells"""
        intensity = settings.get('intensity', 0.5)

        # Cell density mainly affects the signal magnitude
        # For lower density, the overall signal will be lower
        density_factor = max(0.3, 1.0 - intensity * np.random.random())

        # Find baseline
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])
        baseline = np.mean(response[:baseline_end]) if baseline_end > 0 else response[0]

        # Scale response while preserving baseline
        scaled_response = baseline + (response - baseline) * density_factor

        return scaled_response

    def _apply_reagent_stability_issues(self, response, well, row, col, params, settings):
        """Simulate degraded or unstable reagents"""
        intensity = settings.get('intensity', 0.5)

        # Reagent issues mainly affect the peak response
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

        if len(response) <= baseline_end:
            return response

        # Get baseline
        baseline = np.mean(response[:baseline_end])

        # Degraded reagents lead to reduced response
        degradation_factor = max(0.1, 1.0 - intensity * np.random.random())

        # Apply reduction to post-stimulation portion
        altered_response = response.copy()
        altered_response[baseline_end:] = baseline + (response[baseline_end:] - baseline) * degradation_factor

        return altered_response

    def _apply_incorrect_concentrations(self, response, well, row, col, params, settings):
        """Simulate incorrect concentrations of agonists"""
        intensity = settings.get('intensity', 0.5)
        concentration_error = np.random.choice([-1, 1]) * intensity * np.random.random()

        # Find baseline and post-stimulation portions
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

        if len(response) <= baseline_end:
            return response

        # Get baseline
        baseline = np.mean(response[:baseline_end])

        # Apply concentration error effect
        # Positive error -> higher concentration -> stronger response
        # Negative error -> lower concentration -> weaker response
        concentration_factor = 1.0 + concentration_error
        concentration_factor = max(0.1, concentration_factor)  # Ensure it's not too low

        altered_response = response.copy()
        altered_response[baseline_end:] = baseline + (response[baseline_end:] - baseline) * concentration_factor

        return altered_response

    def _apply_reagent_contamination(self, response, well, row, col, params, settings):
        """Simulate contaminated reagents causing unexpected responses"""
        intensity = settings.get('intensity', 0.5)
        contamination_type = np.random.choice(['noise', 'early_response', 'delayed_response'])

        altered_response = response.copy()
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

        if contamination_type == 'noise':
            # Add additional noise to the signal
            noise_level = intensity * 100  # Scale based on intensity
            altered_response += np.random.normal(0, noise_level, size=len(response))

        elif contamination_type == 'early_response':
            # Simulate early activation before agonist addition
            if baseline_end > 10:
                early_start = max(0, baseline_end - int(10 / params['time_interval']))
                baseline = np.mean(altered_response[:early_start])

                # Create a mini-response before the main one
                early_peak = baseline + intensity * (np.max(response) - baseline) * 0.3

                # Generate small peak
                for i in range(early_start, baseline_end):
                    pos = (i - early_start) / (baseline_end - early_start)
                    altered_response[i] = baseline + (early_peak - baseline) * np.sin(pos * np.pi)

        elif contamination_type == 'delayed_response':
            # Simulate delayed or irregular activation
            if len(response) > baseline_end:
                # Delay the response by a random amount
                delay_points = int(intensity * 20 / params['time_interval'])
                if baseline_end + delay_points < len(response):
                    altered_response[baseline_end:baseline_end+delay_points] = response[baseline_end]
                    altered_response[baseline_end+delay_points:] = response[baseline_end:-delay_points] if delay_points < len(response) - baseline_end else response[baseline_end:]

        return np.maximum(altered_response, 0)  # Ensure non-negative values

    def _apply_camera_errors(self, response, well, row, col, params, settings):
        """Simulate camera artifacts and errors"""
        intensity = settings.get('intensity', 0.5)
        error_type = np.random.choice(['dead_pixels', 'saturation', 'noise_spike', 'signal_drop'])

        altered_response = response.copy()

        if error_type == 'dead_pixels':
            # Simulate dead pixels by setting random points to fixed values
            num_dead_pixels = int(intensity * 10)  # Scale with intensity
            for _ in range(num_dead_pixels):
                pixel_pos = random.randint(0, len(response) - 1)
                dead_value = random.choice([0, np.max(response)])
                altered_response[pixel_pos] = dead_value

        elif error_type == 'saturation':
            # Simulate camera saturation at high intensities
            saturation_level = np.max(response) * (1 + 0.2 * intensity)
            altered_response = np.minimum(altered_response, saturation_level)

            # Make the saturated values perfectly flat
            saturated_points = np.where(altered_response >= saturation_level * 0.99)[0]
            altered_response[saturated_points] = saturation_level

        elif error_type == 'noise_spike':
            # Add occasional large spikes of noise
            num_spikes = int(intensity * 5) + 1
            for _ in range(num_spikes):
                spike_pos = random.randint(0, len(response) - 1)
                spike_magnitude = np.max(response) * intensity * 2
                spike_direction = random.choice([-1, 1])
                altered_response[spike_pos] += spike_direction * spike_magnitude

        elif error_type == 'signal_drop':
            # Simulate temporary signal drops
            drop_start = random.randint(0, len(response) - 10)
            drop_length = int(intensity * 10) + 1
            drop_end = min(drop_start + drop_length, len(response))

            # Reduce signal during the drop period
            drop_factor = max(0.1, 1 - intensity * 0.9)
            altered_response[drop_start:drop_end] *= drop_factor

        return np.maximum(altered_response, 0)  # Ensure non-negative values

    def _apply_liquid_handler_issues(self, response, well, row, col, params, settings):
        """Simulate liquid handler issues like inaccurate dispensing"""
        intensity = settings.get('intensity', 0.5)
        issue_type = np.random.choice(['volume_error', 'timing_error', 'no_addition', 'double_addition'])

        altered_response = response.copy()
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

        if len(response) <= baseline_end:
            return response

        if issue_type == 'volume_error':
            # Simulate wrong volume dispensed
            volume_factor = 1.0 + np.random.normal(0, intensity)
            volume_factor = max(0.1, volume_factor)  # Ensure it's not too low

            # Apply to post-addition portion
            baseline = np.mean(response[:baseline_end])
            altered_response[baseline_end:] = baseline + volume_factor * (response[baseline_end:] - baseline)

        elif issue_type == 'timing_error':
            # Simulate addition at wrong time
            timing_shift = int(intensity * 20 / params['time_interval']) * np.random.choice([-1, 1])

            # Ensure we don't shift outside array bounds
            new_baseline_end = max(5, min(len(response) - 10, baseline_end + timing_shift))

            # Create shifted response
            new_response = np.zeros_like(response)
            new_response[:new_baseline_end] = np.mean(response[:baseline_end])

            # Copy the post-addition response shape, but shifted
            post_baseline_length = min(len(response) - new_baseline_end, len(response) - baseline_end)
            if post_baseline_length > 0:
                new_response[new_baseline_end:new_baseline_end+post_baseline_length] = response[baseline_end:baseline_end+post_baseline_length]

            altered_response = new_response

        elif issue_type == 'no_addition':
            # Simulate failed addition - just keep baseline
            baseline = np.mean(response[:baseline_end])
            altered_response[baseline_end:] = baseline + np.random.normal(0, 20, size=len(response[baseline_end:]))

        elif issue_type == 'double_addition':
            # Simulate double addition - add a second peak
            if len(response) > baseline_end + 20:
                second_addition = baseline_end + int(20 / params['time_interval'])

                # Copy the original response shape for the second addition
                second_peak_length = min(len(response) - second_addition, len(response) - baseline_end)

                if second_peak_length > 0:
                    # Add the second response on top of the first
                    baseline = np.mean(response[:baseline_end])
                    second_response = baseline + 0.7 * (response[baseline_end:baseline_end+second_peak_length] - baseline)
                    altered_response[second_addition:second_addition+second_peak_length] += second_response - baseline

        return altered_response

    def _apply_timing_inconsistencies(self, response, well, row, col, params, settings):
        """Simulate timing issues with data collection"""
        intensity = settings.get('intensity', 0.5)
        issue_type = np.random.choice(['missing_points', 'irregular_sampling'])

        altered_response = response.copy()

        if issue_type == 'missing_points':
            # Simulate missing data points
            num_missing = int(intensity * 10) + 1
            for _ in range(num_missing):
                missing_pos = random.randint(0, len(response) - 1)

                # Replace with interpolated nearby points or zeros
                if missing_pos > 0 and missing_pos < len(response) - 1:
                    altered_response[missing_pos] = (altered_response[missing_pos-1] + altered_response[missing_pos+1]) / 2
                else:
                    altered_response[missing_pos] = 0

        elif issue_type == 'irregular_sampling':
            # Simulate irregular time intervals by interpolating at incorrect positions
            num_irregularities = int(intensity * 10) + 1

            for _ in range(num_irregularities):
                # Select a region to distort that's large enough for interpolation
                region_start = random.randint(0, len(response) - 10)
                region_length = random.randint(5, 9)  # Ensure at least 5 points for interpolation
                region_end = min(region_start + region_length, len(response))

                # Check if region is large enough
                if region_end - region_start < 4:
                    continue  # Skip this iteration if region too small

                # Create irregular pattern by interpolating original values at different positions
                original_segment = response[region_start:region_end].copy()

                # Create regular and irregular position arrays
                original_positions = np.linspace(0, 1, len(original_segment))

                # Create irregular sampling pattern (ensure it's strictly increasing)
                jitter = intensity * 0.3 * (np.random.random(len(original_segment)) - 0.5)
                new_positions = original_positions + jitter
                new_positions[0] = 0  # Ensure endpoints remain fixed
                new_positions[-1] = 1
                new_positions = np.sort(new_positions)  # Ensure strictly increasing

                # Ensure minimum distance between points to avoid interpolation issues
                min_distance = 1e-5
                for i in range(1, len(new_positions)):
                    if new_positions[i] - new_positions[i-1] < min_distance:
                        new_positions[i] = new_positions[i-1] + min_distance

                # Interpolate using the appropriate method based on number of points
                try:
                    if len(original_segment) >= 4:
                        # Use cubic interpolation for 4+ points
                        interp_func = interp1d(original_positions, original_segment, kind='cubic',
                                             bounds_error=False, fill_value="extrapolate")
                    elif len(original_segment) >= 3:
                        # Use quadratic interpolation for 3 points
                        interp_func = interp1d(original_positions, original_segment, kind='quadratic',
                                             bounds_error=False, fill_value="extrapolate")
                    else:
                        # Use linear interpolation for 2 points
                        interp_func = interp1d(original_positions, original_segment, kind='linear',
                                             bounds_error=False, fill_value="extrapolate")

                    # Apply interpolation
                    altered_response[region_start:region_end] = interp_func(new_positions)
                except Exception:
                    # Fallback to linear interpolation if other methods fail
                    interp_func = interp1d(original_positions, original_segment, kind='linear',
                                         bounds_error=False, fill_value="extrapolate")
                    altered_response[region_start:region_end] = interp_func(new_positions)

        return altered_response

    def _apply_focus_problems(self, response, well, row, col, params, settings):
        """Simulate focus problems affecting signal quality"""
        intensity = settings.get('intensity', 0.5)

        # Focus issues reduce signal quality and increase noise
        altered_response = response.copy()

        # Add increased noise
        focus_noise = intensity * 50  # Scale based on intensity
        altered_response += np.random.normal(0, focus_noise, size=len(response))

        # Reduce signal contrast (difference between baseline and peak)
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])
        if len(response) > baseline_end:
            baseline = np.mean(response[:baseline_end])
            contrast_factor = max(0.3, 1.0 - intensity * 0.7)

            # Apply contrast reduction while preserving baseline
            altered_response[baseline_end:] = baseline + contrast_factor * (response[baseline_end:] - baseline)

        return np.maximum(altered_response, 0)  # Ensure non-negative values

    def _apply_edge_effects(self, response, well, row, col, params, settings):
        """Simulate edge effects where wells at plate edges show different behavior"""
        intensity = settings.get('intensity', 0.5)

        # Edge effects are stronger for wells at the very edge of the plate
        is_edge = (row == 0 or row == 7 or col == 0 or col == 11)  # For 96-well plate
        is_corner = (row == 0 and col == 0) or (row == 0 and col == 11) or (row == 7 and col == 0) or (row == 7 and col == 11)

        # Determine edge effect magnitude based on position
        if is_corner:
            edge_factor = intensity * 0.8  # Strongest in corners
        elif is_edge:
            edge_factor = intensity * 0.5  # Strong at edges
        else:
            # Declining effect as you move inward
            distance_from_edge = min(row, 7-row, col, 11-col)
            if distance_from_edge <= 1:
                edge_factor = intensity * 0.3  # Moderate for second row/column
            elif distance_from_edge <= 2:
                edge_factor = intensity * 0.1  # Mild for third row/column
            else:
                edge_factor = 0  # No effect for interior wells

        if edge_factor == 0:
            return response  # No change for interior wells

        # Edge effects typically cause:
        # 1. Higher evaporation (higher baseline)
        # 2. Temperature differences (altered kinetics)
        # 3. Reduced signal (due to optical effects)

        altered_response = response.copy()

        # Baseline shift (evaporation effects)
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])
        if baseline_end > 0:
            baseline = np.mean(response[:baseline_end])

            # Increase baseline
            baseline_shift = 1.0 + edge_factor * 0.5
            altered_response[:baseline_end] *= baseline_shift

        # Altered kinetics
        if len(response) > baseline_end:
            # Reduce peak height and change kinetics
            signal_reduction = 1.0 - edge_factor * 0.3
            kinetics_factor = 1.0 + edge_factor * 0.4  # Faster decay

            # Apply to post-baseline portion
            post_baseline = altered_response[baseline_end:]
            baseline = np.mean(altered_response[:baseline_end]) if baseline_end > 0 else altered_response[0]

            # Scale the peak height
            post_baseline = baseline + (post_baseline - baseline) * signal_reduction

            # Alter decay kinetics
            if len(post_baseline) > 5:
                peak_pos = np.argmax(post_baseline)
                if peak_pos < len(post_baseline) - 5:
                    # Apply faster decay after peak
                    decay_portion = post_baseline[peak_pos:]
                    new_decay = baseline + (decay_portion[0] - baseline) * np.exp(
                        -kinetics_factor * np.arange(len(decay_portion)) / len(decay_portion)
                    )
                    post_baseline[peak_pos:] = new_decay

            altered_response[baseline_end:] = post_baseline

        return altered_response

    def _apply_temperature_gradient(self, response, well, row, col, params, settings):
        """Simulate temperature gradient effects across the plate"""
        intensity = settings.get('intensity', 0.5)

        # Define a temperature gradient across the plate
        # For example, cooler on left side, warmer on right
        x_gradient = col / 11.0  # 0 to 1 across columns
        y_gradient = row / 7.0   # 0 to 1 across rows

        # Combine to create a 2D gradient (could be customized for different patterns)
        gradient_pattern = settings.get('pattern', 'left-to-right')

        if gradient_pattern == 'left-to-right':
            temp_factor = x_gradient
        elif gradient_pattern == 'top-to-bottom':
            temp_factor = y_gradient
        elif gradient_pattern == 'center-to-edge':
            center_x, center_y = 5.5, 3.5  # Center of 96-well plate
            distance = np.sqrt((col - center_x)**2 + (row - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            temp_factor = distance / max_distance
        else:
            # Default: diagonal gradient
            temp_factor = (x_gradient + y_gradient) / 2

        # Temperature affects reaction kinetics
        temp_effect = (temp_factor - 0.5) * intensity  # -0.5 to 0.5 times intensity

        altered_response = response.copy()
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

        if len(response) <= baseline_end:
            return response

        # Get baseline
        baseline = np.mean(response[:baseline_end]) if baseline_end > 0 else response[0]
        post_baseline = altered_response[baseline_end:]

        # Warmer areas have:
        # - Faster reaction kinetics (earlier peak)
        # - Slightly higher peak
        # - Faster decay

        # Find original peak
        if len(post_baseline) > 5:
            peak_pos = np.argmax(post_baseline)
            peak_value = post_baseline[peak_pos]

            # Shift peak position based on temperature
            shift_amount = int(temp_effect * 10)  # Shift earlier or later
            new_peak_pos = max(0, min(len(post_baseline) - 1, peak_pos - shift_amount))

            # Adjust peak height
            peak_adjustment = 1.0 + temp_effect * 0.2

            # Create new response curve with shifted peak
            new_post_baseline = np.zeros_like(post_baseline)

            # Create rising phase
            for i in range(new_peak_pos + 1):
                if peak_pos > 0:
                    original_fraction = i / peak_pos
                else:
                    original_fraction = 1.0
                original_fraction = min(1.0, original_fraction)
                new_post_baseline[i] = baseline + original_fraction * (peak_value * peak_adjustment - baseline)

            # Create decay phase
            if new_peak_pos < len(post_baseline) - 1:
                decay_rate = 1.0 + temp_effect * 0.3  # Warmer means faster decay
                decay_length = len(post_baseline) - new_peak_pos - 1

                for i in range(decay_length):
                    fraction = i / decay_length
                    new_post_baseline[new_peak_pos + 1 + i] = baseline + (peak_value * peak_adjustment - baseline) * np.exp(-decay_rate * fraction)

            altered_response[baseline_end:] = new_post_baseline

        return altered_response

    def _apply_evaporation(self, response, well, row, col, params, settings):
        """Simulate effects of evaporation over time"""
        intensity = settings.get('intensity', 0.5)

        # Evaporation is stronger for:
        # - Edge wells (especially corners)
        # - Longer experiments
        # - Later timepoints

        is_edge = (row == 0 or row == 7 or col == 0 or col == 11)  # For 96-well plate
        is_corner = (row == 0 and col == 0) or (row == 0 and col == 11) or (row == 7 and col == 0) or (row == 7 and col == 11)

        if is_corner:
            position_factor = 1.0
        elif is_edge:
            position_factor = 0.7
        else:
            # Declining effect as you move inward
            distance_from_edge = min(row, 7-row, col, 11-col)
            position_factor = max(0.1, 1.0 - distance_from_edge * 0.3)

        # Evaporation effects increase over time
        time_points = len(response)
        altered_response = response.copy()

        for i in range(time_points):
            # Increasing effect over time
            time_factor = i / time_points

            # Evaporation causes:
            # 1. Concentration of dye (higher fluorescence)
            # 2. Potential photobleaching compensation

            # Combine factors
            evaporation_effect = 1.0 + position_factor * intensity * time_factor * 0.4

            altered_response[i] *= evaporation_effect

        return altered_response

    def _apply_well_crosstalk(self, response, well, row, col, params, settings):
        """Simulate optical crosstalk between adjacent wells"""
        intensity = settings.get('intensity', 0.5)

        # Simplistic model: assume some percentage of signal bleeds from neighboring wells
        # In a real implementation, this would depend on the actual contents of adjacent wells

        # Generate a random "neighbor" signal
        neighbor_response = np.random.random(len(response)) * np.max(response) * 0.8

        # Add a realistic calcium response shape to the neighbor
        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])
        if len(neighbor_response) > baseline_end + 5:
            # Create a peak
            baseline = np.mean(response[:baseline_end]) if baseline_end > 0 else 0
            peak_pos = baseline_end + random.randint(5, 20)
            peak_height = baseline + np.random.random() * np.max(response) * 0.5

            # Create response curve
            for i in range(baseline_end):
                neighbor_response[i] = baseline

            for i in range(baseline_end, peak_pos):
                fraction = (i - baseline_end) / (peak_pos - baseline_end)
                neighbor_response[i] = baseline + fraction * (peak_height - baseline)

            for i in range(peak_pos, len(neighbor_response)):
                decay_fraction = (i - peak_pos) / (len(neighbor_response) - peak_pos)
                neighbor_response[i] = baseline + (peak_height - baseline) * np.exp(-decay_fraction * 3)

        # Add some percentage of the neighbor signal to this well
        crosstalk_factor = intensity * 0.1  # Max 10% crosstalk with max intensity
        altered_response = response + crosstalk_factor * neighbor_response

        return altered_response

    def _calculate_dose_response(self, agonist, concentration):
        """
        Calculate dose-response factor using Hill equation

        Args:
            agonist (str): Agonist name
            concentration (float): Concentration in µM

        Returns:
            float: Response factor between 0 and 1
        """
        # EC50 and Hill coefficient values for different agonists
        ec50_values = {
            'ATP': 100.0,  # EC50 of 100 µM
            'UTP': 150.0,  # EC50 of 150 µM
            'Ionomycin': 0.5,  # EC50 of 0.5 µM
            'Buffer': float('inf')  # No response
        }

        hill_coefficients = {
            'ATP': 1.5,
            'UTP': 1.3,
            'Ionomycin': 2.0,
            'Buffer': 1.0
        }

        # Get EC50 and Hill coefficient, or use defaults if not found
        ec50 = ec50_values.get(agonist, 100.0)
        hill = hill_coefficients.get(agonist, 1.0)

        # Handle special case for Buffer (should give minimal response)
        if agonist == 'Buffer' or concentration <= 0:
            return 0.05  # Minimal response for buffer

        # Calculate response using Hill equation
        response = concentration**hill / (ec50**hill + concentration**hill)

        # Ensure response is between 0 and 1
        return max(0, min(1, response))

    # new method to handle custom errors
    def _apply_custom_error(self, response, well, row, col, params, settings):
        """Apply a custom error with specific parameters to a response"""
        error_type = settings.get('custom_type', 'random_spikes')
        error_params = settings.get('custom_params', {})
        use_global = settings.get('use_global_settings', False)
        altered_response = response.copy()

        # Apply global intensity if specified
        if use_global and 'intensity' in params:
            intensity = params.get('intensity', 0.5)
            # Update relevant parameters that depend on intensity
            if error_type == 'random_spikes':
                error_params['amplitude'] = error_params.get('amplitude', 1000) * intensity
            elif error_type == 'dropouts':
                error_params['factor'] = max(0.01, error_params.get('factor', 0.1) * (1 - intensity))
            elif error_type == 'baseline_drift':
                error_params['magnitude'] = error_params.get('magnitude', 500) * intensity
            elif error_type == 'oscillating_baseline':
                error_params['amplitude'] = error_params.get('amplitude', 300) * intensity
            elif error_type == 'signal_cutout':
                error_params['duration_pct'] = min(0.9, error_params.get('duration_pct', 0.2) * intensity)
            elif error_type == 'incomplete_decay':
                error_params['elevation_factor'] = error_params.get('elevation_factor', 0.5) * intensity
            elif error_type == 'extra_noise':
                error_params['std'] = error_params.get('std', 100) * intensity
            elif error_type == 'overlapping_oscillation':
                error_params['amplitude'] = error_params.get('amplitude', 200) * intensity
            elif error_type == 'sudden_jump':
                error_params['magnitude'] = error_params.get('magnitude', 500) * intensity
            elif error_type == 'exponential_drift':
                error_params['magnitude'] = error_params.get('magnitude', 1000) * intensity
            elif error_type == 'delayed_response':
                error_params['delay_seconds'] = min(20, error_params.get('delay_seconds', 5) * intensity)


        if error_type == 'random_spikes':
            # Add random spikes to the trace
            num_spikes = error_params.get('num_spikes', 3)
            spike_amplitude = error_params.get('amplitude', 1000)
            spike_width = error_params.get('width', 3)

            for _ in range(num_spikes):
                # Choose random position for spike
                pos = random.randint(0, len(response) - spike_width - 1)

                # Create spike (Gaussian shape)
                for i in range(spike_width):
                    # Calculate distance from center of spike (in range -1 to 1)
                    distance = 2 * (i - spike_width/2) / spike_width
                    # Apply Gaussian factor
                    factor = np.exp(-4 * distance**2)
                    # Add spike
                    altered_response[pos + i] += spike_amplitude * factor

        elif error_type == 'dropouts':
            # Add signal dropouts (periods of zero or reduced signal)
            num_dropouts = error_params.get('num_dropouts', 2)
            dropout_length = error_params.get('length', 10)
            dropout_factor = error_params.get('factor', 0.1)  # How much signal remains

            for _ in range(num_dropouts):
                # Choose random position for dropout
                pos = random.randint(0, len(response) - dropout_length - 1)

                # Apply dropout
                for i in range(dropout_length):
                    altered_response[pos + i] = response[pos + i] * dropout_factor

        elif error_type == 'baseline_drift':
            # Add rising or falling baseline drift
            drift_direction = error_params.get('direction', 'rising')
            drift_magnitude = error_params.get('magnitude', 500)

            # Calculate drift factor for each timepoint
            drift = np.zeros_like(response)
            for i in range(len(response)):
                drift_factor = i / len(response)  # 0 to 1
                if drift_direction == 'rising':
                    drift[i] = drift_magnitude * drift_factor
                else:  # falling
                    drift[i] = -drift_magnitude * drift_factor

            # Add drift to response
            altered_response += drift

        elif error_type == 'oscillating_baseline':
            # Add oscillating baseline
            frequency = error_params.get('frequency', 0.05)  # cycles per timepoint
            amplitude = error_params.get('amplitude', 300)

            # Calculate oscillation
            time = np.arange(len(response))
            oscillation = amplitude * np.sin(2 * np.pi * frequency * time)

            # Add oscillation
            altered_response += oscillation

        elif error_type == 'signal_cutout':
            # Completely remove signal for a period
            cutout_start = error_params.get('start_pct', 0.3)  # percentage of trace length
            cutout_duration = error_params.get('duration_pct', 0.2)  # percentage of trace length

            start_idx = int(cutout_start * len(response))
            duration = int(cutout_duration * len(response))
            end_idx = min(start_idx + duration, len(response))

            # Replace with baseline or zeros
            if error_params.get('replace_with_baseline', True):
                # Use average of first few points as baseline
                baseline = np.mean(response[:min(10, len(response))])
                altered_response[start_idx:end_idx] = baseline
            else:
                altered_response[start_idx:end_idx] = 0

        elif error_type == 'incomplete_decay':
            # Response doesn't return to baseline
            baseline_end = int(params.get('agonist_addition_time', 10) /
                             params.get('time_interval', 0.4))

            if len(response) > baseline_end + 10:
                # Find peak after baseline
                post_addition = response[baseline_end:]
                peak_idx = np.argmax(post_addition)
                decay_start_idx = baseline_end + peak_idx

                if decay_start_idx < len(response) - 10:
                    # Calculate baseline
                    baseline = np.mean(response[:baseline_end]) if baseline_end > 0 else response[0]

                    # Get elevation factor (how much above baseline the signal stays)
                    elevation_factor = error_params.get('elevation_factor', 0.5)

                    # Get peak height
                    peak_height = response[decay_start_idx] - baseline

                    # Calculate incomplete decay
                    for i in range(decay_start_idx, len(response)):
                        # Original value
                        orig_value = response[i]
                        # How far the original value is from baseline
                        distance_from_baseline = orig_value - baseline
                        # Add elevation (more for points closer to baseline)
                        factor = 1 - (distance_from_baseline / peak_height)
                        elevation = peak_height * elevation_factor * max(0, factor)
                        altered_response[i] = orig_value + elevation

        elif error_type == 'extra_noise':
            # Add extra Gaussian noise to signal
            noise_std = error_params.get('std', 100)

            # Generate noise
            noise = np.random.normal(0, noise_std, len(response))

            # Add noise to signal
            altered_response += noise

        elif error_type == 'overlapping_oscillation':
            # Add an oscillating signal (like a sine wave) that might represent interference
            frequency = error_params.get('frequency', 0.1)  # cycles per timepoint
            amplitude = error_params.get('amplitude', 200)
            phase_shift = error_params.get('phase_shift', 0)  # in radians

            # Calculate oscillation
            time = np.arange(len(response))
            oscillation = amplitude * np.sin(2 * np.pi * frequency * time + phase_shift)

            # Add oscillation
            altered_response += oscillation

        elif error_type == 'sudden_jump':
            # Add a sudden jump in the signal
            jump_position = error_params.get('position_pct', 0.7)  # percentage of trace length
            jump_magnitude = error_params.get('magnitude', 500)

            # Calculate jump position
            jump_idx = int(jump_position * len(response))

            # Apply jump to all points after the position
            altered_response[jump_idx:] += jump_magnitude

        elif error_type == 'exponential_drift':
            # Add exponential drift to baseline
            direction = error_params.get('direction', 'upward')
            magnitude = error_params.get('magnitude', 1000)
            rate = error_params.get('rate', 3)  # Higher values = faster exponential change

            # Calculate exponential drift
            time = np.arange(len(response)) / len(response)  # 0 to 1
            if direction == 'upward':
                drift = magnitude * (np.exp(rate * time) - 1) / (np.exp(rate) - 1)
            else:  # downward
                drift = magnitude * (np.exp(rate * (1 - time)) - 1) / (np.exp(rate) - 1)

            # Add drift
            altered_response += drift

        elif error_type == 'delayed_response':
            # Delay the calcium response by a certain amount
            delay_time = error_params.get('delay_seconds', 5)  # seconds
            baseline_end = int(params.get('agonist_addition_time', 10) /
                             params.get('time_interval', 0.4))

            if len(response) > baseline_end:
                # Calculate delay in timepoints
                delay_points = int(delay_time / params.get('time_interval', 0.4))

                # Get baseline value
                baseline = np.mean(response[:baseline_end]) if baseline_end > 0 else response[0]

                # Create new response with delay
                new_response = np.zeros_like(response)
                new_response[:baseline_end + delay_points] = baseline

                # Copy the response shape but shifted
                remaining_points = len(response) - (baseline_end + delay_points)
                if remaining_points > 0:
                    points_to_copy = min(remaining_points, len(response) - baseline_end)
                    new_response[baseline_end + delay_points:baseline_end + delay_points + points_to_copy] = \
                        response[baseline_end:baseline_end + points_to_copy]

                altered_response = new_response

        return altered_response

