import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger('FLIPR_Simulator.ConfigManager')

class ConfigManager:
    """Manages configuration for FLIPR simulator, including saving/loading settings"""

    def __init__(self, config_dir='config'):
        """Initialize the configuration manager"""
        self.config_dir = config_dir
        self.setup_dir = os.path.join(config_dir, 'setup_files')
        self.presets_dir = os.path.join(config_dir, 'presets')

        # Create config directories if they don't exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.setup_dir, exist_ok=True)
        os.makedirs(self.presets_dir, exist_ok=True)

        # Default configuration
        self.default_config = {
            'general': {
                'plate_type': '96-well',
                'num_timepoints': 451,
                'time_interval': 0.4,
                'agonist_addition_time': 10,
                'random_seed': 42
            },
            'noise': {
                'read_noise': 20,
                'background': 100,
                'photobleaching_rate': 0.0005
            },
            'export': {
                'output_dir': 'simulation_results',
                'format': 'csv'
            },
            'cell_lines': {
                'Positive Control': {
                    'baseline': 500,
                    'peak_ionomycin': 4000,
                    'peak_other': 1000,
                    'rise_rate_ionomycin': 0.12,
                    'rise_rate_other': 0.10,
                    'decay_rate_ionomycin': 0.07,
                    'decay_rate_other': 0.06
                },
                'Negative Control': {
                    'baseline': 500,
                    'peak_ionomycin': 3800,
                    'peak_other': 325,
                    'rise_rate_ionomycin': 0.12,
                    'rise_rate_other': 0.08,
                    'decay_rate_ionomycin': 0.02,
                    'decay_rate_other': 0.01
                },
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
                    'baseline': 100,            # Lower baseline (just background)
                    'peak_ionomycin': 110,      # Minimal change from baseline
                    'peak_other': 105,          # Even less change for other agonists
                    'rise_rate_ionomycin': 0.01, # Very slow rise
                    'rise_rate_other': 0.01,     # Very slow rise
                    'decay_rate_ionomycin': 0.01, # Very slow decay
                    'decay_rate_other': 0.01      # Very slow decay
                },
            },
            'agonists': {
                'ATP': 2.0,
                'UTP': 1.8,
                'Ionomycin': 3.0,
                'Buffer': 0.1
            },
            'errors': {
                'cell_variability': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'dye_loading': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'cell_health': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'cell_density': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'reagent_stability': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'reagent_concentration': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'reagent_contamination': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'camera_errors': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'liquid_handler': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'timing_errors': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'focus_problems': {'active': False, 'probability': 0.2, 'intensity': 0.5},
                'edge_effects': {'active': False, 'probability': 0.5, 'intensity': 0.5},
                'temperature_gradient': {'active': False, 'probability': 1.0, 'intensity': 0.5, 'pattern': 'left-to-right'},
                'evaporation': {'active': False, 'probability': 0.5, 'intensity': 0.5},
                'well_crosstalk': {'active': False, 'probability': 0.2, 'intensity': 0.5}
            }
        }

        # Current configuration (starts as a copy of default)
        self.config = self.default_config.copy()

        logger.info("Configuration manager initialized")

    def save_config(self, filename, config=None):
        """
        Save the current or specified configuration to file

        Args:
            filename (str): Name of the file to save the configuration to
            config (dict, optional): Configuration to save. If None, saves current config.

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            if config is None:
                config = self.config

            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'

            filepath = os.path.join(self.config_dir, filename)

            # Add timestamp to saved config
            save_config = config.copy()
            save_config['_meta'] = {
                'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': '1.0'
            }

            with open(filepath, 'w') as f:
                json.dump(save_config, f, indent=4)

            logger.info(f"Configuration saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration to {filename}: {str(e)}", exc_info=True)
            return False

    def load_config(self, filename):
        """
        Load configuration from file

        Args:
            filename (str): Name of the file to load configuration from

        Returns:
            dict: Loaded configuration or None if loading failed
        """
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'

            filepath = os.path.join(self.config_dir, filename)

            if not os.path.exists(filepath):
                logger.error(f"Configuration file {filepath} not found")
                return None

            with open(filepath, 'r') as f:
                loaded_config = json.load(f)

            # Remove metadata if present
            if '_meta' in loaded_config:
                del loaded_config['_meta']

            logger.info(f"Configuration loaded from {filepath}")

            # Update current configuration
            self.config = loaded_config

            return loaded_config

        except Exception as e:
            logger.error(f"Error loading configuration from {filename}: {str(e)}", exc_info=True)
            return None

    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config = self.default_config.copy()
        logger.info("Configuration reset to defaults")
        return self.config

    def update_config(self, section, key, value):
        """
        Update a specific configuration setting

        Args:
            section (str): Configuration section
            key (str): Configuration key
            value: New value for the setting

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            if section not in self.config:
                self.config[section] = {}

            self.config[section][key] = value
            logger.info(f"Updated configuration: {section}.{key} = {value}")
            return True

        except Exception as e:
            logger.error(f"Error updating configuration {section}.{key}: {str(e)}", exc_info=True)
            return False

    def get_config_value(self, section, key, default=None):
        """
        Get a specific configuration value

        Args:
            section (str): Configuration section
            key (str): Configuration key
            default: Default value to return if the key is not found

        Returns:
            The configuration value or the default if not found
        """
        try:
            return self.config.get(section, {}).get(key, default)
        except Exception as e:
            logger.error(f"Error getting configuration {section}.{key}: {str(e)}", exc_info=True)
            return default

    def get_active_errors(self):
        """
        Get a dictionary of active error models

        Returns:
            dict: Dictionary of active error models and their settings
        """
        active_errors = {}

        for error_type, settings in self.config.get('errors', {}).items():
            if settings.get('active', False):
                active_errors[error_type] = settings

        return active_errors

    def get_simulation_config(self):
        """
        Get a consolidated configuration dictionary for simulation

        Returns:
            dict: Configuration suitable for passing to the simulation engine
        """
        # Extract the relevant parts of the configuration for simulation
        sim_config = {
            'num_timepoints': self.config['general']['num_timepoints'],
            'time_interval': self.config['general']['time_interval'],
            'agonist_addition_time': self.config['general']['agonist_addition_time'],
            'random_seed': self.config['general']['random_seed'],
            'read_noise': self.config['noise']['read_noise'],
            'background': self.config['noise']['background'],
            'photobleaching_rate': self.config['noise']['photobleaching_rate'],
            'cell_lines': self.config['cell_lines'],
            'agonists': self.config['agonists'],
            'active_errors': self.get_active_errors()
        }

        # Add plate dimensions based on plate type
        if self.config['general']['plate_type'] == '96-well':
            sim_config['num_wells'] = 96
            sim_config['rows'] = 8
            sim_config['cols'] = 12
        elif self.config['general']['plate_type'] == '384-well':
            sim_config['num_wells'] = 384
            sim_config['rows'] = 16
            sim_config['cols'] = 24

        return sim_config

    def save_plate_layout(self, layout_name, layout_data, overwrite=False):
        """
        Save a plate layout to a CSV file

        Args:
            layout_name (str): Name of the layout (e.g., 'agonist_layout', 'cell_line_layout')
            layout_data (pd.DataFrame or np.array): Layout data to save
            overwrite (bool): Whether to overwrite existing file

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Ensure filename has .csv extension
            if not layout_name.endswith('.csv'):
                layout_name += '.csv'

            filepath = os.path.join(self.setup_dir, layout_name)

            if os.path.exists(filepath) and not overwrite:
                logger.warning(f"Layout file {filepath} already exists, use overwrite=True to overwrite")
                return False

            # Convert numpy array to DataFrame if necessary
            if isinstance(layout_data, np.ndarray):
                # Create row and column labels for 96-well plate
                if layout_data.shape == (8, 12):  # 96-well plate
                    rows = list('ABCDEFGH')
                    cols = [str(i) for i in range(1, 13)]
                elif layout_data.shape == (16, 24):  # 384-well plate
                    rows = list('ABCDEFGHIJKLMNOP')
                    cols = [str(i) for i in range(1, 25)]
                else:
                    rows = range(layout_data.shape[0])
                    cols = range(layout_data.shape[1])

                layout_data = pd.DataFrame(layout_data, index=rows, columns=cols)

            # Save to CSV
            layout_data.to_csv(filepath)

            logger.info(f"Plate layout saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving plate layout to {layout_name}: {str(e)}", exc_info=True)
            return False

    def load_plate_layout(self, layout_name, default_creator=None):
        """
        Load a plate layout from a CSV file

        Args:
            layout_name (str): Name of the layout to load
            default_creator (callable, optional): Function to create default layout if file not found

        Returns:
            pd.DataFrame or np.ndarray: Loaded layout or None if loading failed
        """
        try:
            # Ensure filename has .csv extension
            if not layout_name.endswith('.csv'):
                layout_name += '.csv'

            filepath = os.path.join(self.setup_dir, layout_name)

            if not os.path.exists(filepath):
                logger.warning(f"Layout file {filepath} not found")

                if default_creator is not None:
                    logger.info(f"Creating default layout for {layout_name}")
                    default_layout = default_creator()
                    self.save_plate_layout(layout_name, default_layout, overwrite=True)
                    return default_layout

                return None

            # Load CSV
            layout_data = pd.read_csv(filepath, index_col=0)

            logger.info(f"Plate layout loaded from {filepath}")
            return layout_data

        except Exception as e:
            logger.error(f"Error loading plate layout from {layout_name}: {str(e)}", exc_info=True)
            return None

    def save_preset(self, preset_name, description=""):
        """
        Save current configuration as a preset

        Args:
            preset_name (str): Name of the preset
            description (str): Description of the preset

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Ensure filename has .json extension
            if not preset_name.endswith('.json'):
                preset_name += '.json'

            filepath = os.path.join(self.presets_dir, preset_name)

            # Create preset with metadata
            preset = {
                'config': self.config,
                'meta': {
                    'name': preset_name.replace('.json', ''),
                    'description': description,
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }

            with open(filepath, 'w') as f:
                json.dump(preset, f, indent=4)

            logger.info(f"Preset saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving preset {preset_name}: {str(e)}", exc_info=True)
            return False

    def load_preset(self, preset_name):
        """
        Load a preset configuration

        Args:
            preset_name (str): Name of the preset to load

        Returns:
            dict: Loaded preset configuration or None if loading failed
        """
        try:
            # Ensure filename has .json extension
            if not preset_name.endswith('.json'):
                preset_name += '.json'

            filepath = os.path.join(self.presets_dir, preset_name)

            if not os.path.exists(filepath):
                logger.error(f"Preset file {filepath} not found")
                return None

            with open(filepath, 'r') as f:
                preset = json.load(f)

            # Update current configuration
            if 'config' in preset:
                self.config = preset['config']
                logger.info(f"Preset {preset_name} loaded successfully")
                return self.config
            else:
                logger.error(f"Invalid preset format in {preset_name}")
                return None

        except Exception as e:
            logger.error(f"Error loading preset {preset_name}: {str(e)}", exc_info=True)
            return None

    def list_presets(self):
        """
        List all available presets

        Returns:
            list: List of dictionaries containing preset information
        """
        try:
            presets = []

            for filename in os.listdir(self.presets_dir):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(self.presets_dir, filename)
                        with open(filepath, 'r') as f:
                            preset = json.load(f)

                        if 'meta' in preset:
                            presets.append(preset['meta'])
                        else:
                            presets.append({
                                'name': filename.replace('.json', ''),
                                'description': 'No description available',
                                'created_at': 'Unknown'
                            })
                    except:
                        # If we can't read the preset, just include the filename
                        presets.append({
                            'name': filename.replace('.json', ''),
                            'description': 'Error reading preset',
                            'created_at': 'Unknown'
                        })

            return presets

        except Exception as e:
            logger.error(f"Error listing presets: {str(e)}", exc_info=True)
            return []

    def create_preset_error_scenarios(self):
        """
        Create a set of preset error scenarios for common FLIPR failures

        Returns:
            bool: True if creation was successful, False otherwise
        """
        try:
            # Define preset scenarios
            scenarios = [
                {
                    'name': 'normal_operation',
                    'description': 'Normal operation with no errors',
                    'errors': {}  # All errors inactive
                },
                {
                    'name': 'dye_loading_issues',
                    'description': 'Problems with calcium dye loading in cells',
                    'errors': {
                        'dye_loading': {'active': True, 'probability': 0.4, 'intensity': 0.7}
                    }
                },
                {
                    'name': 'cell_health_problems',
                    'description': 'Issues with cell health affecting calcium responses',
                    'errors': {
                        'cell_health': {'active': True, 'probability': 0.4, 'intensity': 0.7}
                    }
                },
                {
                    'name': 'liquid_handler_failure',
                    'description': 'Problems with agonist addition by liquid handler',
                    'errors': {
                        'liquid_handler': {'active': True, 'probability': 0.5, 'intensity': 0.8}
                    }
                },
                {
                    'name': 'edge_effects',
                    'description': 'Plate edge effects affecting well responses',
                    'errors': {
                        'edge_effects': {'active': True, 'probability': 1.0, 'intensity': 0.8}
                    }
                },
                {
                    'name': 'focus_problems',
                    'description': 'Camera focus issues affecting image quality',
                    'errors': {
                        'focus_problems': {'active': True, 'probability': 0.5, 'intensity': 0.6}
                    }
                },
                {
                    'name': 'reagent_degradation',
                    'description': 'Agonists showing reduced potency due to degradation',
                    'errors': {
                        'reagent_stability': {'active': True, 'probability': 0.8, 'intensity': 0.6}
                    }
                },
                {
                    'name': 'camera_failure',
                    'description': 'Intermittent camera errors during acquisition',
                    'errors': {
                        'camera_errors': {'active': True, 'probability': 0.4, 'intensity': 0.7}
                    }
                },
                {
                    'name': 'temperature_effects',
                    'description': 'Temperature gradient across plate affecting kinetics',
                    'errors': {
                        'temperature_gradient': {'active': True, 'probability': 1.0, 'intensity': 0.7, 'pattern': 'left-to-right'}
                    }
                },
                {
                    'name': 'timing_problems',
                    'description': 'Inconsistent timing in data acquisition',
                    'errors': {
                        'timing_errors': {'active': True, 'probability': 0.3, 'intensity': 0.5}
                    }
                },
                {
                    'name': 'combined_failures',
                    'description': 'Multiple failures occurring simultaneously',
                    'errors': {
                        'edge_effects': {'active': True, 'probability': 0.8, 'intensity': 0.5},
                        'liquid_handler': {'active': True, 'probability': 0.3, 'intensity': 0.6},
                        'cell_variability': {'active': True, 'probability': 0.4, 'intensity': 0.4}
                    }
                }
            ]

            # Create each preset
            for scenario in scenarios:
                # Start with default config
                config = self.default_config.copy()

                # Update errors with scenario settings
                for error_type, settings in scenario['errors'].items():
                    if error_type in config['errors']:
                        config['errors'][error_type].update(settings)

                # Save as preset
                preset = {
                    'config': config,
                    'meta': {
                        'name': scenario['name'],
                        'description': scenario['description'],
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'category': 'error_scenario'
                    }
                }

                preset_path = os.path.join(self.presets_dir, f"{scenario['name']}.json")
                with open(preset_path, 'w') as f:
                    json.dump(preset, f, indent=4)

                logger.info(f"Created preset scenario: {scenario['name']}")

            return True

        except Exception as e:
            logger.error(f"Error creating preset scenarios: {str(e)}", exc_info=True)
            return False

    def create_demo_cell_lines(self):
        """
        Create a set of demo cell lines with different calcium response characteristics

        Returns:
            bool: True if creation was successful, False otherwise
        """
        try:
            # Define cell lines with varying calcium response characteristics
            demo_cell_lines = {
                'Fast_Responder': {
                    'baseline': 500,
                    'peak_ionomycin': 4200,
                    'peak_other': 1200,
                    'rise_rate': 0.15,  # Fast rise
                    'decay_rate': 0.08   # Moderate decay
                },
                'Slow_Responder': {
                    'baseline': 500,
                    'peak_ionomycin': 3800,
                    'peak_other': 900,
                    'rise_rate': 0.05,  # Slow rise
                    'decay_rate': 0.03   # Slow decay
                },
                'High_Amplitude': {
                    'baseline': 450,
                    'peak_ionomycin': 4500,
                    'peak_other': 1500,
                    'rise_rate': 0.10,
                    'decay_rate': 0.05
                },
                'Low_Amplitude': {
                    'baseline': 550,
                    'peak_ionomycin': 3000,
                    'peak_other': 700,
                    'rise_rate': 0.10,
                    'decay_rate': 0.05
                },
                'Fast_Decay': {
                    'baseline': 500,
                    'peak_ionomycin': 4000,
                    'peak_other': 1000,
                    'rise_rate': 0.10,
                    'decay_rate': 0.10   # Fast decay
                },
                'Sustained_Response': {
                    'baseline': 500,
                    'peak_ionomycin': 4000,
                    'peak_other': 1000,
                    'rise_rate': 0.10,
                    'decay_rate': 0.01   # Very slow decay (sustained)
                },
                'High_Baseline': {
                    'baseline': 800,     # High baseline
                    'peak_ionomycin': 4000,
                    'peak_other': 1000,
                    'rise_rate': 0.10,
                    'decay_rate': 0.05
                },
                'Low_Baseline': {
                    'baseline': 300,     # Low baseline
                    'peak_ionomycin': 3800,
                    'peak_other': 1000,
                    'rise_rate': 0.10,
                    'decay_rate': 0.05
                }
            }

            # Update current config with these cell lines
            self.config['cell_lines'].update(demo_cell_lines)

            # Save as a preset
            preset = {
                'config': self.config,
                'meta': {
                    'name': 'demo_cell_lines',
                    'description': 'Demonstration cell lines with varied calcium response characteristics',
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'category': 'cell_lines'
                }
            }

            preset_path = os.path.join(self.presets_dir, 'demo_cell_lines.json')
            with open(preset_path, 'w') as f:
                json.dump(preset, f, indent=4)

            logger.info("Created demo cell lines preset")
            return True

        except Exception as e:
            logger.error(f"Error creating demo cell lines: {str(e)}", exc_info=True)
            return False

    def create_demo_agonists(self):
        """
        Create a set of demo agonists with different response characteristics

        Returns:
            bool: True if creation was successful, False otherwise
        """
        try:
            # Define a range of agonists with different potencies
            demo_agonists = {
                'ATP': 2.0,              # Standard
                'UTP': 1.8,              # Slightly weaker than ATP
                'Carbachol': 1.5,        # Moderate activator
                'Glutamate': 1.2,        # Weak activator
                'Histamine': 1.3,        # Moderate activator
                'Bradykinin': 1.7,       # Strong activator
                'Thrombin': 1.9,         # Very strong activator
                'Ionomycin': 3.0,        # Calcium ionophore (maximal)
                'DMSO': 0.05,            # Vehicle control
                'Buffer': 0.1,           # Buffer control
            }

            # Update current config with these agonists
            self.config['agonists'].update(demo_agonists)

            # Save as a preset
            preset = {
                'config': self.config,
                'meta': {
                    'name': 'demo_agonists',
                    'description': 'Demonstration agonists with varied response potencies',
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'category': 'agonists'
                }
            }

            preset_path = os.path.join(self.presets_dir, 'demo_agonists.json')
            with open(preset_path, 'w') as f:
                json.dump(preset, f, indent=4)

            logger.info("Created demo agonists preset")
            return True

        except Exception as e:
            logger.error(f"Error creating demo agonists: {str(e)}", exc_info=True)
            return False
