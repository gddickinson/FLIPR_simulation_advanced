import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import seaborn as sns
import logging
import os
from datetime import datetime

logger = logging.getLogger('FLIPR_Simulator.Visualization')

class PlotManager:
    """Manages visualization of FLIPR simulation data"""

    def __init__(self):
        """Initialize the plot manager"""
        # Default color maps for different cell lines and agonists
        self.cell_line_colors = {
            'Positive Control': '#E41A1C',  # Red
            'Negative Control': '#377EB8',  # Blue
            'Neurotypical': '#4DAF4A',      # Green
            'ASD': '#984EA3',               # Purple
            'FXS': '#FF7F00',               # Orange
            'NTC': '#999999',               # Gray
            'Default': '#666666'            # Darker Gray
        }

        self.agonist_colors = {
            'ATP': '#66C2A5',               # Teal
            'UTP': '#FC8D62',               # Orange
            'Carbachol': '#8DA0CB',         # Blue
            'Ionomycin': '#E78AC3',         # Pink
            'Buffer': '#A6D854',            # Green
            'Default': '#999999'            # Gray
        }

        # Initialize figure counters
        self.figure_counter = 0

    def get_color(self, item, color_dict):
        """Get color for an item from a color dictionary, with fallback to default"""
        return color_dict.get(item, color_dict.get('Default', '#999999'))

    def create_all_traces_plot(self, simulation_results, figure=None, ax=None):
        """
        Create a plot showing all response traces in the plate

        Args:
            simulation_results (dict): Results from simulation
            figure (Figure, optional): Matplotlib figure to use
            ax (Axes, optional): Matplotlib axes to use

        Returns:
            Figure: Matplotlib figure containing the plot
        """
        # Check if we should use DF/F0 data
        use_df_f0 = simulation_results['params'].get('simulate_df_f0', False) and 'df_f0_data' in simulation_results

        #DEBUG
        logger.info(f"Visualization - Using DF/F0 data: {use_df_f0}")
        if use_df_f0:
            logger.info(f"DF/F0 data available: {'df_f0_data' in simulation_results}")
            if 'df_f0_data' in simulation_results:
                logger.info(f"DF/F0 data shape: {simulation_results['df_f0_data'].shape}")
                logger.info(f"DF/F0 display as percentage: {simulation_results['params'].get('df_f0_as_percent', True)}")


        if use_df_f0:
            # Use DF/F0 data
            plate_data = simulation_results['df_f0_data']
            y_label = 'ΔF/F₀ (%)' if simulation_results['params'].get('df_f0_as_percent', True) else 'ΔF/F₀ (ratio)'
        else:
            # Use raw fluorescence data
            plate_data = simulation_results['plate_data']
            y_label = 'Fluorescence (A.U.)'

        metadata = simulation_results['metadata']
        time_points = simulation_results['time_points']

        # Create figure if not provided
        if figure is None or ax is None:
            figure = Figure(figsize=(10, 6), dpi=100)
            ax = figure.add_subplot(111)

        # Plot each well's trace
        for well_idx, well_data in enumerate(plate_data):
            if well_idx < len(metadata):
                well_meta = metadata[well_idx]

                # Only plot if the well is valid
                if well_meta.get('valid', True):
                    cell_line = well_meta.get('cell_line', 'Default')
                    color = self.get_color(cell_line, self.cell_line_colors)

                    ax.plot(time_points, well_data, color=color, alpha=0.3,
                            linewidth=0.8, label=f"_{cell_line}")  # Underscore for grouping in legend

        # Create a custom legend that shows one entry per cell line
        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l.startswith('_'):  # Remove the underscore prefix used for grouping
                clean_label = l[1:]
                if clean_label not in by_label:
                    by_label[clean_label] = h

        legend_handles = [by_label[label] for label in by_label]
        legend_labels = list(by_label.keys())

        ax.legend(legend_handles, legend_labels, loc='upper right')

        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(y_label)

        # Add appropriate title based on data type
        if use_df_f0:
            df_f0_type = "%" if simulation_results['params'].get('df_f0_as_percent', True) else "ratio"
            ax.set_title(f'All Calcium Response Traces (ΔF/F₀ {df_f0_type})')
        else:
            ax.set_title('All Calcium Response Traces')

        figure.tight_layout()
        return figure

    def create_heatmap(self, simulation_results, timepoint_index=None, figure=None, ax=None):
        """
        Create a heatmap of the plate at a specific timepoint

        Args:
            simulation_results (dict): Results from simulation
            timepoint_index (int, optional): Index of timepoint to display. If None, use peak value for each well.
            figure (Figure, optional): Matplotlib figure to use
            ax (Axes, optional): Matplotlib axes to use

        Returns:
            Figure: Matplotlib figure containing the heatmap
        """
        # Check if we should use DF/F0 data
        use_df_f0 = simulation_results['params'].get('simulate_df_f0', False) and 'df_f0_data' in simulation_results

        if use_df_f0:
            plate_data = simulation_results['df_f0_data']
            colorbar_label_prefix = 'ΔF/F₀'
            if simulation_results['params'].get('df_f0_as_percent', True):
                colorbar_label_suffix = ' (%)'
            else:
                colorbar_label_suffix = ' (ratio)'
        else:
            plate_data = simulation_results['plate_data']
            colorbar_label_prefix = 'Fluorescence'
            colorbar_label_suffix = ''


        metadata = simulation_results['metadata']
        time_points = simulation_results['time_points']
        params = simulation_results['params']

        # Get plate dimensions
        rows = params.get('rows', 8)
        cols = params.get('cols', 12)

        # Create figure if not provided
        if figure is None or ax is None:
            figure = Figure(figsize=(12, 8), dpi=100)
            ax = figure.add_subplot(111)

        # Prepare data for heatmap
        heatmap_data = np.zeros((rows, cols))

        for well_idx, well_data in enumerate(plate_data):
            if well_idx < len(metadata):
                row, col = well_idx // cols, well_idx % cols
                if row < rows and col < cols:
                    # Get value for heatmap
                    if timepoint_index is not None:
                        # Use specified timepoint
                        if timepoint_index < len(well_data):
                            heatmap_data[row, col] = well_data[timepoint_index]
                    else:
                        # Use peak value
                        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])
                        if len(well_data) > baseline_end:
                            baseline = np.mean(well_data[:baseline_end]) if baseline_end > 0 else well_data[0]
                            # Find peak after baseline
                            peak_value = np.max(well_data[baseline_end:])
                            heatmap_data[row, col] = peak_value - baseline

        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='viridis', interpolation='nearest')

        # Add colorbar
        cbar = figure.colorbar(im, ax=ax)
        # Update colorbar label
        if timepoint_index is not None:
            cbar.set_label(f'{colorbar_label_prefix} at t={time_points[timepoint_index]:.1f}s{colorbar_label_suffix}')
        else:
            if use_df_f0:
                cbar.set_label(f'Peak Response {colorbar_label_prefix}{colorbar_label_suffix}')
            else:
                cbar.set_label('Peak Response (F-F0)')

        # Add well labels
        for i in range(rows):
            for j in range(cols):
                well_id = f"{chr(65 + i)}{j + 1}"
                ax.text(j, i, well_id, ha='center', va='center', color='white', fontsize=8)

        # Set ticks and labels
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(range(1, cols + 1))
        ax.set_yticklabels([chr(65 + i) for i in range(rows)])

        # Set title
        if timepoint_index is not None:
            ax.set_title(f'Plate Heatmap at t={time_points[timepoint_index]:.1f}s')
        else:
            ax.set_title('Peak Response Heatmap')

        figure.tight_layout()
        return figure

    def create_cell_line_comparison(self, simulation_results, figure=None, ax=None):
        """
        Create a plot comparing different cell lines

        Args:
            simulation_results (dict): Results from simulation
            figure (Figure, optional): Matplotlib figure to use
            ax (Axes, optional): Matplotlib axes to use

        Returns:
            Figure: Matplotlib figure containing the plot
        """
        # Check if we should use DF/F0 data
        use_df_f0 = simulation_results['params'].get('simulate_df_f0', False) and 'df_f0_data' in simulation_results

        if use_df_f0:
            plate_data = simulation_results['df_f0_data']
            y_label = 'ΔF/F₀ (%)' if simulation_results['params'].get('df_f0_as_percent', True) else 'ΔF/F₀ (ratio)'
        else:
            plate_data = simulation_results['plate_data']
            y_label = 'Fluorescence (A.U.)'


        metadata = simulation_results['metadata']
        time_points = simulation_results['time_points']
        params = simulation_results['params']

        # Group data by cell line
        cell_line_data = {}

        for well_idx, well_data in enumerate(plate_data):
            if well_idx < len(metadata):
                well_meta = metadata[well_idx]

                if well_meta.get('valid', True):
                    cell_line = well_meta.get('cell_line', 'Unknown')
                    agonist = well_meta.get('agonist', 'Unknown')

                    # Create nested dictionary by cell line and agonist
                    if cell_line not in cell_line_data:
                        cell_line_data[cell_line] = {}

                    if agonist not in cell_line_data[cell_line]:
                        cell_line_data[cell_line][agonist] = []

                    cell_line_data[cell_line][agonist].append(well_data)

        # Count number of cell lines for determining subplot layout
        num_cell_lines = len(cell_line_data)

        # Create figure if not provided
        if figure is None:
            if num_cell_lines <= 2:
                figure = Figure(figsize=(12, 6), dpi=100)
            else:
                # Calculate rows/columns for subplots
                n_cols = min(2, num_cell_lines)
                n_rows = (num_cell_lines + n_cols - 1) // n_cols
                figure = Figure(figsize=(12, 6 * n_rows), dpi=100)

        # Create subplots if axes not provided
        if ax is None:
            if num_cell_lines == 1:
                ax = [figure.add_subplot(111)]
            else:
                ax = figure.subplots(nrows=(num_cell_lines + 1) // 2, ncols=min(2, num_cell_lines))
                if num_cell_lines == 2:
                    ax = [ax[0], ax[1]]  # Convert to list for consistent indexing

        # Ensure ax is always a list
        if not isinstance(ax, list) and not isinstance(ax, np.ndarray):
            ax = [ax]

        # Plot each cell line in its own subplot
        for i, (cell_line, agonist_data) in enumerate(cell_line_data.items()):
            # Get the appropriate subplot
            if i < len(ax):
                current_ax = ax[i]

                # Plot each agonist for this cell line
                for agonist, well_list in agonist_data.items():
                    if len(well_list) > 0:
                        # Convert to numpy array for easier manipulation
                        well_array = np.array(well_list)

                        # Calculate mean and std
                        mean_response = np.mean(well_array, axis=0)
                        std_response = np.std(well_array, axis=0)

                        # Get color for this agonist
                        color = self.get_color(agonist, self.agonist_colors)

                        # Plot mean response
                        current_ax.plot(time_points, mean_response, label=agonist, color=color)

                        # Add std deviation area
                        current_ax.fill_between(time_points,
                                              mean_response - std_response,
                                              mean_response + std_response,
                                              color=color, alpha=0.2)

                # Set title and labels
                current_ax.set_title(f'Cell Line: {cell_line}')
                current_ax.set_xlabel('Time (s)')
                current_ax.set_ylabel(y_label)
                current_ax.legend()

        figure.tight_layout()
        return figure

    def create_agonist_comparison(self, simulation_results, figure=None, ax=None):
        """
        Create a plot comparing different agonists

        Args:
            simulation_results (dict): Results from simulation
            figure (Figure, optional): Matplotlib figure to use
            ax (Axes, optional): Matplotlib axes to use

        Returns:
            Figure: Matplotlib figure containing the plot
        """
        # Check if we should use DF/F0 data
        use_df_f0 = simulation_results['params'].get('simulate_df_f0', False) and 'df_f0_data' in simulation_results

        if use_df_f0:
            plate_data = simulation_results['df_f0_data']
            y_label = 'ΔF/F₀ (%)' if simulation_results['params'].get('df_f0_as_percent', True) else 'ΔF/F₀ (ratio)'
        else:
            plate_data = simulation_results['plate_data']
            y_label = 'Fluorescence (A.U.)'

        metadata = simulation_results['metadata']
        time_points = simulation_results['time_points']
        params = simulation_results['params']

        # Group data by agonist
        agonist_data = {}

        for well_idx, well_data in enumerate(plate_data):
            if well_idx < len(metadata):
                well_meta = metadata[well_idx]

                if well_meta.get('valid', True):
                    cell_line = well_meta.get('cell_line', 'Unknown')
                    agonist = well_meta.get('agonist', 'Unknown')

                    # Create nested dictionary by agonist and cell line
                    if agonist not in agonist_data:
                        agonist_data[agonist] = {}

                    if cell_line not in agonist_data[agonist]:
                        agonist_data[agonist][cell_line] = []

                    agonist_data[agonist][cell_line].append(well_data)

        # Count number of agonists for determining subplot layout
        num_agonists = len(agonist_data)

        # Create figure if not provided
        if figure is None:
            if num_agonists <= 2:
                figure = Figure(figsize=(12, 6), dpi=100)
            else:
                # Calculate rows/columns for subplots
                n_cols = min(2, num_agonists)
                n_rows = (num_agonists + n_cols - 1) // n_cols
                figure = Figure(figsize=(12, 6 * n_rows), dpi=100)

        # Create subplots if axes not provided
        if ax is None:
            if num_agonists == 1:
                ax = [figure.add_subplot(111)]
            else:
                ax = figure.subplots(nrows=(num_agonists + 1) // 2, ncols=min(2, num_agonists))
                if num_agonists == 2:
                    ax = [ax[0], ax[1]]  # Convert to list for consistent indexing

        # Ensure ax is always a list
        if not isinstance(ax, list) and not isinstance(ax, np.ndarray):
            ax = [ax]

        # Plot each agonist in its own subplot
        for i, (agonist, cell_line_data) in enumerate(agonist_data.items()):
            # Get the appropriate subplot
            if i < len(ax):
                current_ax = ax[i]

                # Plot each cell line for this agonist
                for cell_line, well_list in cell_line_data.items():
                    if len(well_list) > 0:
                        # Convert to numpy array for easier manipulation
                        well_array = np.array(well_list)

                        # Calculate mean and std
                        mean_response = np.mean(well_array, axis=0)
                        std_response = np.std(well_array, axis=0)

                        # Get color for this cell line
                        color = self.get_color(cell_line, self.cell_line_colors)

                        # Plot mean response
                        current_ax.plot(time_points, mean_response, label=cell_line, color=color)

                        # Add std deviation area
                        current_ax.fill_between(time_points,
                                              mean_response - std_response,
                                              mean_response + std_response,
                                              color=color, alpha=0.2)

                # Set title and labels
                current_ax.set_title(f'Agonist: {agonist}')
                current_ax.set_xlabel('Time (s)')
                current_ax.set_ylabel(y_label)
                current_ax.legend()

        figure.tight_layout()
        return figure

    def create_peak_response_boxplot(self, simulation_results, figure=None, ax=None):
        """
        Create a boxplot of peak responses grouped by cell line and agonist

        Args:
            simulation_results (dict): Results from simulation
            figure (Figure, optional): Matplotlib figure to use
            ax (Axes, optional): Matplotlib axes to use

        Returns:
            Figure: Matplotlib figure containing the boxplot
        """
        # Check if we should use DF/F0 data
        use_df_f0 = simulation_results['params'].get('simulate_df_f0', False) and 'df_f0_data' in simulation_results

        if use_df_f0:
            plate_data = simulation_results['df_f0_data']
            y_label = 'Peak Response ΔF/F₀'
            if simulation_results['params'].get('df_f0_as_percent', True):
                y_label += ' (%)'
            else:
                y_label += ' (ratio)'
        else:
            plate_data = simulation_results['plate_data']
            y_label = 'Peak Response (F-F0)'



        metadata = simulation_results['metadata']
        params = simulation_results['params']

        # Prepare data for boxplot
        boxplot_data = []
        labels = []
        colors = []

        # Group by cell line and agonist
        cell_line_agonist_peaks = {}

        baseline_end = int(params['agonist_addition_time'] / params['time_interval'])

        for well_idx, well_data in enumerate(plate_data):
            if well_idx < len(metadata):
                well_meta = metadata[well_idx]

                if well_meta.get('valid', True):
                    cell_line = well_meta.get('cell_line', 'Unknown')
                    agonist = well_meta.get('agonist', 'Unknown')

                    # Calculate peak response
                    if len(well_data) > baseline_end:
                        baseline = np.mean(well_data[:baseline_end]) if baseline_end > 0 else well_data[0]
                        peak_value = np.max(well_data[baseline_end:])
                        peak_response = peak_value - baseline

                        # Group key
                        key = f"{cell_line} - {agonist}"

                        if key not in cell_line_agonist_peaks:
                            cell_line_agonist_peaks[key] = {
                                'peaks': [],
                                'cell_line': cell_line,
                                'agonist': agonist
                            }

                        cell_line_agonist_peaks[key]['peaks'].append(peak_response)

        # Convert to format suitable for boxplot
        for key, data in cell_line_agonist_peaks.items():
            boxplot_data.append(data['peaks'])
            labels.append(key)
            colors.append(self.get_color(data['cell_line'], self.cell_line_colors))

        # Create figure if not provided
        if figure is None or ax is None:
            figure = Figure(figsize=(12, 8), dpi=100)
            ax = figure.add_subplot(111)

        # Create boxplot
        if boxplot_data:
            bplot = ax.boxplot(boxplot_data, patch_artist=True, labels=labels)

            # Color boxes by cell line
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Set labels and title
            ax.set_xlabel('Cell Line - Agonist')
            ax.set_ylabel(y_label)
            ax.set_title('Peak Response by Cell Line and Agonist')

            # Rotate x-axis labels for better readability
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No valid data for boxplot',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)

        figure.tight_layout()
        return figure

    def create_well_detail_plot(self, simulation_results, well_id, figure=None, ax=None):
        """
        Create a detailed plot for a specific well

        Args:
            simulation_results (dict): Results from simulation
            well_id (str): Well ID (e.g., 'A1')
            figure (Figure, optional): Matplotlib figure to use
            ax (Axes, optional): Matplotlib axes to use

        Returns:
            Figure: Matplotlib figure containing the plot
        """
        # Check if we should use DF/F0 data
        use_df_f0 = simulation_results['params'].get('simulate_df_f0', False) and 'df_f0_data' in simulation_results

        if use_df_f0:
            plate_data = simulation_results['df_f0_data']
            y_label = 'ΔF/F₀ (%)' if simulation_results['params'].get('df_f0_as_percent', True) else 'ΔF/F₀ (ratio)'
        else:
            plate_data = simulation_results['plate_data']
            y_label = 'Fluorescence (A.U.)'

        metadata = simulation_results['metadata']
        time_points = simulation_results['time_points']
        params = simulation_results['params']

        # Create figure if not provided
        if figure is None or ax is None:
            figure = Figure(figsize=(10, 6), dpi=100)
            ax = figure.add_subplot(111)

        # Find the well index
        well_idx = None
        for i, meta in enumerate(metadata):
            if meta.get('well_id') == well_id:
                well_idx = i
                break

        if well_idx is not None and well_idx < len(plate_data):
            well_data = plate_data[well_idx]
            well_meta = metadata[well_idx]

            # Plot data
            ax.plot(time_points, well_data, color='blue', label='Response')

            # Mark agonist addition time
            agonist_time = params['agonist_addition_time']
            ax.axvline(x=agonist_time, color='red', linestyle='--',
                     label=f'Agonist Addition ({agonist_time}s)')

            # Calculate and mark baseline and peak
            baseline_end = int(agonist_time / params['time_interval'])

            if len(well_data) > baseline_end:
                # Calculate baseline
                baseline = np.mean(well_data[:baseline_end]) if baseline_end > 0 else well_data[0]
                ax.axhline(y=baseline, color='green', linestyle=':',
                         label=f'Baseline ({baseline:.2f})')

                # Find and mark peak
                post_baseline = well_data[baseline_end:]
                peak_idx = baseline_end + np.argmax(post_baseline)
                peak_value = well_data[peak_idx]
                peak_time = time_points[peak_idx]

                ax.plot(peak_time, peak_value, 'ro', markersize=8,
                      label=f'Peak ({peak_value:.2f})')

                # Add peak response annotation
                peak_response = peak_value - baseline
                ax.annotate(f'Peak Response: {peak_response:.2f}',
                          xy=(peak_time, peak_value),
                          xytext=(peak_time + 5, peak_value + 200),
                          arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

            # Set title and labels
            cell_line = well_meta.get('cell_line', 'Unknown')
            agonist = well_meta.get('agonist', 'Unknown')

            ax.set_title(f'Well {well_id}: {cell_line} with {agonist}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(y_label)
            ax.legend()

        else:
            ax.text(0.5, 0.5, f'Well {well_id} not found or invalid',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)

        figure.tight_layout()
        return figure

    def create_plate_layout_plot(self, cell_line_layout, agonist_layout, figure=None):
        """
        Create a visualization of the plate layout showing cell lines and agonists

        Args:
            cell_line_layout (np.ndarray): Layout of cell lines
            agonist_layout (np.ndarray): Layout of agonists
            figure (Figure, optional): Matplotlib figure to use

        Returns:
            Figure: Matplotlib figure containing the layout plot
        """
        # Create figure if not provided
        if figure is None:
            figure = Figure(figsize=(12, 6), dpi=100)

        # Create subplots
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)

        # Get plate dimensions
        rows, cols = cell_line_layout.shape if cell_line_layout is not None else (8, 12)

        # Plot cell line layout
        if cell_line_layout is not None:
            # Create a mapping of cell lines to numeric values for coloring
            unique_cell_lines = np.unique(cell_line_layout)
            cell_line_map = {cell_line: i for i, cell_line in enumerate(unique_cell_lines)}

            # Create a numeric representation for the heatmap
            cell_line_numeric = np.zeros_like(cell_line_layout, dtype=float)
            for i in range(rows):
                for j in range(cols):
                    cell_line_numeric[i, j] = cell_line_map.get(cell_line_layout[i, j], -1)

            # Plot heatmap
            im1 = ax1.imshow(cell_line_numeric, cmap='viridis', interpolation='nearest')

            # Add well labels
            for i in range(rows):
                for j in range(cols):
                    well_id = f"{chr(65 + i)}{j + 1}"
                    cell_line = cell_line_layout[i, j]
                    # Truncate long cell line names
                    if len(str(cell_line)) > 6:
                        cell_line = str(cell_line)[:6] + '..'
                    text = f"{well_id}\n{cell_line}"
                    ax1.text(j, i, text, ha='center', va='center', color='white', fontsize=8)

            # Create custom legend
            legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                                        markerfacecolor=self.get_color(cell_line, self.cell_line_colors),
                                        markersize=10, label=cell_line)
                             for cell_line in unique_cell_lines]

            ax1.legend(handles=legend_elements, title="Cell Lines",
                     loc='upper center', bbox_to_anchor=(0.5, -0.1))

            # Set title and labels
            ax1.set_title('Cell Line Layout')
            ax1.set_xticks(np.arange(cols))
            ax1.set_yticks(np.arange(rows))
            ax1.set_xticklabels(range(1, cols + 1))
            ax1.set_yticklabels([chr(65 + i) for i in range(rows)])

        # Plot agonist layout
        if agonist_layout is not None:
            # Create a mapping of agonists to numeric values for coloring
            unique_agonists = np.unique(agonist_layout)
            agonist_map = {agonist: i for i, agonist in enumerate(unique_agonists)}

            # Create a numeric representation for the heatmap
            agonist_numeric = np.zeros_like(agonist_layout, dtype=float)
            for i in range(rows):
                for j in range(cols):
                    agonist_numeric[i, j] = agonist_map.get(agonist_layout[i, j], -1)

            # Plot heatmap
            im2 = ax2.imshow(agonist_numeric, cmap='plasma', interpolation='nearest')

            # Add well labels
            for i in range(rows):
                for j in range(cols):
                    well_id = f"{chr(65 + i)}{j + 1}"
                    agonist = agonist_layout[i, j]
                    # Truncate long agonist names
                    if len(str(agonist)) > 6:
                        agonist = str(agonist)[:6] + '..'
                    text = f"{well_id}\n{agonist}"
                    ax2.text(j, i, text, ha='center', va='center', color='white', fontsize=8)

            # Create custom legend
            legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                                        markerfacecolor=self.get_color(agonist, self.agonist_colors),
                                        markersize=10, label=agonist)
                             for agonist in unique_agonists]

            ax2.legend(handles=legend_elements, title="Agonists",
                     loc='upper center', bbox_to_anchor=(0.5, -0.1))

            # Set title and labels
            ax2.set_title('Agonist Layout')
            ax2.set_xticks(np.arange(cols))
            ax2.set_yticks(np.arange(rows))
            ax2.set_xticklabels(range(1, cols + 1))
            ax2.set_yticklabels([chr(65 + i) for i in range(rows)])

        figure.tight_layout()
        return figure

    def create_animation_frames(self, simulation_results, output_dir='animation_frames'):
        """
        Create a series of heatmap frames for animation

        Args:
            simulation_results (dict): Results from simulation
            output_dir (str): Directory to save frames

        Returns:
            bool: True if frames were created successfully
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Get parameters
            time_points = simulation_results['time_points']
            num_timepoints = len(time_points)

            # Create frames (not every timepoint, to keep file count reasonable)
            frame_interval = max(1, num_timepoints // 50)  # At most 50 frames

            for i in range(0, num_timepoints, frame_interval):
                # Create heatmap for this timepoint
                fig = Figure(figsize=(8, 6), dpi=100)
                ax = fig.add_subplot(111)

                self.create_heatmap(simulation_results, timepoint_index=i, figure=fig, ax=ax)

                # Save frame
                filename = os.path.join(output_dir, f'frame_{i:04d}.png')
                fig.savefig(filename)
                plt.close(fig)

                logger.info(f"Created animation frame {i+1}/{num_timepoints}")

            return True

        except Exception as e:
            logger.error(f"Error creating animation frames: {str(e)}", exc_info=True)
            return False

    def create_error_comparison_plot(self, normal_results, error_results, error_type, figure=None):
        """
        Create a plot comparing normal response with error-affected response

        Args:
            normal_results (dict): Simulation results without errors
            error_results (dict): Simulation results with errors
            error_type (str): Type of error being compared
            figure (Figure, optional): Matplotlib figure to use

        Returns:
            Figure: Matplotlib figure containing the comparison plot
        """
        # Create figure if not provided
        if figure is None:
            figure = Figure(figsize=(12, 8), dpi=100)

        # Create subplots for different views
        gs = figure.add_gridspec(2, 2)
        ax1 = figure.add_subplot(gs[0, 0])  # Example well comparison
        ax2 = figure.add_subplot(gs[0, 1])  # Peak response comparison
        ax3 = figure.add_subplot(gs[1, :])  # Overall traces

        # Get time points
        time_points = normal_results['time_points']

        # Find a well affected by the error
        normal_data = normal_results['plate_data']
        error_data = error_results['plate_data']
        metadata = normal_results['metadata']  # Assuming same metadata

        # Find largest difference between normal and error data
        max_diff_idx = 0
        max_diff = 0

        for i in range(len(normal_data)):
            if i < len(metadata) and metadata[i].get('valid', True):
                diff = np.sum(np.abs(normal_data[i] - error_data[i]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = i

        # Plot example well comparison
        if max_diff_idx < len(normal_data) and max_diff_idx < len(error_data):
            ax1.plot(time_points, normal_data[max_diff_idx], 'b-', label='Normal')
            ax1.plot(time_points, error_data[max_diff_idx], 'r-', label='With Error')

            if max_diff_idx < len(metadata):
                well_id = metadata[max_diff_idx].get('well_id', f'Well {max_diff_idx}')
                ax1.set_title(f'{error_type} Effect: {well_id}')
            else:
                ax1.set_title(f'{error_type} Effect: Example Well')

            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Fluorescence (A.U.)')
            ax1.legend()

        # Calculate peak responses for both datasets
        baseline_end = int(normal_results['params']['agonist_addition_time'] /
                          normal_results['params']['time_interval'])

        normal_peaks = []
        error_peaks = []

        for i in range(len(normal_data)):
            if i < len(metadata) and metadata[i].get('valid', True) and i < len(error_data):
                # Calculate normal peak
                if len(normal_data[i]) > baseline_end:
                    n_baseline = np.mean(normal_data[i][:baseline_end]) if baseline_end > 0 else normal_data[i][0]
                    n_peak = np.max(normal_data[i][baseline_end:])
                    normal_peaks.append(n_peak - n_baseline)

                    # Calculate error peak
                    e_baseline = np.mean(error_data[i][:baseline_end]) if baseline_end > 0 else error_data[i][0]
                    e_peak = np.max(error_data[i][baseline_end:])
                    error_peaks.append(e_peak - e_baseline)

        # Plot peak response comparison
        ax2.boxplot([normal_peaks, error_peaks], labels=['Normal', 'With Error'])
        ax2.set_title('Peak Response Comparison')
        ax2.set_ylabel('Peak Response (F-F0)')

        # Calculate percent difference
        if len(normal_peaks) > 0 and len(error_peaks) > 0:
            normal_mean = np.mean(normal_peaks)
            error_mean = np.mean(error_peaks)
            percent_diff = ((error_mean - normal_mean) / normal_mean) * 100

            ax2.annotate(f'Mean Difference: {percent_diff:.1f}%',
                       xy=(1.5, max(np.max(normal_peaks), np.max(error_peaks))),
                       xytext=(0, -20), textcoords='offset points',
                       ha='center')

        # Plot all traces
        for i in range(len(normal_data)):
            if i < len(metadata) and metadata[i].get('valid', True) and i < len(error_data):
                ax3.plot(time_points, normal_data[i], 'b-', alpha=0.2)
                ax3.plot(time_points, error_data[i], 'r-', alpha=0.2)

        # Add legend lines with higher opacity
        ax3.plot([], [], 'b-', label='Normal', alpha=0.8)
        ax3.plot([], [], 'r-', label='With Error', alpha=0.8)

        ax3.set_title(f'All Traces: {error_type} Effect')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Fluorescence (A.U.)')
        ax3.legend()

        figure.tight_layout()
        return figure

    def export_plot(self, figure, filename, dpi=300):
        """
        Export a plot to a file

        Args:
            figure (Figure): Matplotlib figure to export
            filename (str): Output filename
            dpi (int): Resolution in dots per inch

        Returns:
            bool: True if export was successful
        """
        try:
            figure.savefig(filename, dpi=dpi, bbox_inches='tight')
            logger.info(f"Plot exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting plot to {filename}: {str(e)}", exc_info=True)
            return False
