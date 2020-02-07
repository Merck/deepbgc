import logging

from deepbgc.output.writer import OutputWriter
from deepbgc import util
from matplotlib import pyplot as plt
import numpy as np
import warnings

class PfamScorePlotWriter(OutputWriter):

    def __init__(self, out_path, max_sequences=50):
        super(PfamScorePlotWriter, self).__init__(out_path)
        self.sequence_scores = []
        self.sequence_titles = []
        self.sequence_thresholds = []
        self.sequence_detector_names = []
        self.max_sequences = max_sequences

    @classmethod
    def get_description(cls):
        return 'BGC detection scores of each Pfam domain in genomic order'

    @classmethod
    def get_name(cls):
        return 'pfam-score-plot'

    def close(self):
        self.save_plot()

    def save_plot(self):
        num_sequences = len(self.sequence_titles)
        if not num_sequences:
            return
        fig, axes = plt.subplots(num_sequences, 1, figsize=(15, 1+1.5*num_sequences))
        if num_sequences == 1:
            axes = [axes]
        offset = 0.05
        for i, (detector_scores, sequence_title, sequence_thresholds) in enumerate(zip(self.sequence_scores, self.sequence_titles, self.sequence_thresholds)):
            axes[i].set_ylim(0-offset, 1+offset)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('BGC score')
            axes[i].set_title(sequence_title)
            x = detector_scores.index.values
            xlim = (min(x), max(x))
            axes[i].set_xlim(xlim)
            if detector_scores.empty:
                continue
            cmap = plt.get_cmap("tab10")
            # For each detector score column
            color_idx = 0
            for column, thresholds in zip(detector_scores.columns, sequence_thresholds):
                y = detector_scores[column].values
                if column == 'in_cluster':
                    if not np.any(y):
                        continue
                    color = 'grey'
                    full_height_val = y * (1 + 2 * offset) - offset
                    axes[i].fill_between(x, full_height_val, -offset, color=color, alpha=0.3)
                    axes[i].step(x, full_height_val, where='post', lw=0.75, alpha=0.75, color=color, label='annotated')
                else:
                    color = cmap(color_idx)
                    color_idx += 1
                    marker = 'o' if len(x) == 1 else None
                    axes[i].plot(x, y, lw=0.75, alpha=0.6, color=color, label=column, marker=marker)
                    axes[i].hlines(thresholds, xlim[0], xlim[1], color=color, linestyles='--', lw=0.75, alpha=0.5)
            if len(detector_scores.columns) > 1:
                lgnd = axes[i].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                for line in lgnd.get_lines():
                    line.set_linewidth(2)

        axes[-1].set_xlabel('Pfam domains in genomic order')
        fig.tight_layout()
        logging.debug('Saving per-pfam BGC score plot to: %s', self.out_path)
        fig.savefig(self.out_path, dpi=150, bbox_inches='tight')

    def write(self, record):
        if self.max_sequences is not None and len(self.sequence_titles) > self.max_sequences:
            warnings.warn('Reached maximum number of {} sequences for plotting, some sequences will not be plotted.'.format(self.max_sequences))
            return
        scores = util.create_pfam_dataframe(record, add_in_cluster=True)
        if scores.empty:
            logging.debug('Skipping score plot for empty record %s', record.id)
            return
        detector_meta = util.get_record_detector_meta(record)
        detector_names = np.unique([meta['name'] for meta in detector_meta.values()])
        score_columns = ['in_cluster'] + [util.format_bgc_score_column(name) for name in detector_names]
        title = record.id
        if record.description and record.description != record.id:
            title = '{} ({})'.format(record.id, record.description)
        # Each model can have multiple labels, each with a different threshold
        thresholds = [None]
        for name in detector_names:
            thresholds.append([float(meta['score_threshold']) for meta in detector_meta.values() if meta['name'] == name])
        self.sequence_scores.append(scores[score_columns])
        self.sequence_titles.append(title)
        self.sequence_thresholds.append(thresholds)
        self.sequence_detector_names.append(detector_names)
