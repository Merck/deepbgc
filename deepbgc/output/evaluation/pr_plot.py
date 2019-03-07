import logging
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from deepbgc.output.evaluation.roc_plot import CurvePlotWriter


class PrecisionRecallPlotWriter(CurvePlotWriter):

    @classmethod
    def get_description(cls):
        return 'Precision-Recall curve based on predicted per-Pfam BGC scores'

    @classmethod
    def get_name(cls):
        return 'pr-plot'

    def plot_curve(self, true_values, predictions, ax=None, title='Precision-Recall', label='PR', lw=1, **kwargs):
        """
        Plot Precision-Recall curve of a single model. Can be called repeatedly with same axis to plot multiple curves.
        :param true_values: Series of true values
        :param predictions: Series of prediction values
        :param ax: Use given axis (will create new one if None)
        :param title: Plot title
        :param label: ROC curve label
        :param lw: Line width
        :param kwargs: Additional arguments for plotting function
        :return: Figure axis
        """
        precision, recall, _ = precision_recall_curve(true_values, predictions)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        avg_precision = average_precision_score(true_values, predictions)
        label_pr = label + ': {:.3f} AvgPrec'.format(avg_precision)
        logging.info('Precision-Recall result: %s', label_pr)

        ax.step(recall, precision, where='post', label=label_pr, lw=lw, **kwargs)

        ax.set_title(title)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.05])
        ax.legend(loc='upper right', bbox_to_anchor=(1, -0.13))

        return ax
