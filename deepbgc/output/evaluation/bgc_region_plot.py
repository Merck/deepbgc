from __future__ import (
    print_function,
    division,
    absolute_import,
)
from deepbgc.output.writer import OutputWriter
from deepbgc import util
from matplotlib import pyplot as plt
import numpy as np
import logging
import warnings

class BGCRegionPlotWriter(OutputWriter):

    def __init__(self, out_path, max_sequences=50):
        super(BGCRegionPlotWriter, self).__init__(out_path)
        self.detector_labels = []
        self.sequence_clusters = []
        self.sequence_titles = []
        self.max_sequences = max_sequences

    @classmethod
    def get_description(cls):
        return 'Detected BGCs plotted by their nucleotide coordinates'

    @classmethod
    def get_name(cls):
        return 'bgc-region-plot'

    def close(self):
        self.save_plot()

    def save_plot(self):
        num_sequences = len(self.sequence_titles)
        num_detectors = len(self.detector_labels)

        fig, axes = plt.subplots(num_sequences, 1, figsize=(15, 1 + 0.25 * (num_detectors + 2) * num_sequences))
        if num_sequences == 1:
            axes = [axes]

        for i, (clusters, sequence_title) in enumerate(zip(self.sequence_clusters, self.sequence_titles)):
            ax = axes[i]
            ax.set_facecolor('white')
            ax.set_yticks(range(1, num_detectors + 1))
            ax.set_yticklabels(reversed(self.detector_labels))
            ax.set_ylim([0.3, num_detectors + 0.7])
            ax.set_title(sequence_title)

            if clusters.empty:
                continue

            end = clusters['nucl_end'].max()
            x_step = 100000
            if end / x_step > 20:
                x_step = 200000
            if end / x_step > 20:
                x_step = 500000

            clusters_by_detector = clusters.groupby('detector_label')
            cmap = plt.get_cmap("tab10")
            color_idx = 0
            for level, detector_label in enumerate(self.detector_labels):
                if detector_label not in clusters_by_detector.groups:
                    continue
                detector_clusters = clusters_by_detector.get_group(detector_label)

                if detector_label.lower() == 'annotated':
                    color = 'grey'
                    for c, cluster in detector_clusters.iterrows():
                        ax.axvspan(cluster['nucl_start'], cluster['nucl_end'], color='black', alpha=0.13)
                else:
                    color = cmap(color_idx)
                    color_idx += 1

                # Get coordinates as a vector cand1_start, cand1_end, cand2_start, cand2_end, ...
                x = detector_clusters[['nucl_start', 'nucl_end']].values.reshape(1, -1)[0]
                # Get y values as a vector y_line_coordinate, nan, y_line_coordinate, nan, ...
                y = np.ones(x.shape) * (num_detectors - level) # 5, 4, 3, 2, 1
                y[1::2] = np.nan
                # Create multiple thin lines to avoid extending actual cluster region due to line width
                for d in np.arange(-0.08, 0.08, 0.005):
                    ax.step(x, y + d, color=color, where='post', lw=0.3, label=None)
            ax.set_xlabel('')
            xticks = range(0, clusters['nucl_end'].max() + x_step, x_step)
            ax.set_xticks(xticks)
            ax.set_xticklabels(['{:.0f}kb'.format(x / 1e3) if x < 1e6 else '{:.1f}Mb'.format(x / 1e6) for x in xticks])
        axes[-1].set_xlabel('Nucleotide coordinates')
        fig.tight_layout()
        logging.debug('Saving BGC region plot to: %s', self.out_path)
        fig.savefig(self.out_path, dpi=150, bbox_inches='tight')

    def write(self, record):
        if len(self.sequence_titles) > self.max_sequences:
            warnings.warn('Reached maximum number of {} sequences for plotting, some sequences will not be plotted.'.format(self.max_sequences))
            return
        clusters = util.create_cluster_dataframe(record, add_pfams=False, add_proteins=False)
        title = record.id
        if record.description and record.description != record.id:
            title = '{} ({})'.format(record.id, record.description)

        if len(clusters):
            self.detector_labels = sorted(np.unique(list(clusters['detector_label'].unique()) + self.detector_labels), key=lambda l: l.lower())
        self.sequence_clusters.append(clusters)
        self.sequence_titles.append(title)
