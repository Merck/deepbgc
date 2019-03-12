from __future__ import (
    print_function,
    division,
    absolute_import,
)
import pandas as pd
import numpy as np
import logging
import sys

class RandomForestClassifier(object):
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier as RFC
        self.rf = RFC(**kwargs)

    def predict(self, X):
        """
        Get BGC class prediction for a set of samples, represented by a DataFrame of sample vectors (e.g. One-hot encoding)
        :param X: DataFrame of sample vectors, one sample per line
        :return: DataFrame of per-class prediction scores, one sample per line
        """
        if isinstance(X, pd.Series):
            # Turn single sample Series into a single-row DataFrame
            X = pd.DataFrame([X])
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Expected DataFrame, got: {}'.format(type(X)))

        if len(X):
            predictions = np.array([class_pred[:,1] for class_pred in self.rf.predict_proba(X)]).T
        else:
            predictions = []
        return pd.DataFrame(predictions, columns=self.targets_, index=X.index)

    def fit(self, X_list, y_list, sample_weights=None, verbose=0, debug_progress_path=None,
            validation_X_list=None, validation_y_list=None):
        """
        Create and train internal HMM model based on list of positive and negative samples.
        Emission probability will be calculated from samples. Starting and transition probability have to be provided.

        :param X_list: DataFrame of samples, one sample per line, each sample represented by a single vector
        :param y_list: DataFrame of sample target vectors
        :param sample_weights: List of sample weights, marking their contribution to the emission probability. If not provided, will be set to 1 for all samples.
        :param verbose: Verbosity (0 = no output, 1 = plot top pfams for positive and negative states)
        :param debug_progress_path: Not used in RF model.
        :param validation_X_list: List of validation samples, not used in RF model.
        :param validation_y_list: List of validation states, not used in RF model.
        :return: self
        """
        if not isinstance(X_list, pd.DataFrame):
            raise ValueError("Random Forest expects samples in a single pd.DataFrame, got {}".format(type(X_list)))
        if not isinstance(y_list, pd.DataFrame):
            raise ValueError("Random Forest expects targets in a single pd.DataFrame, got {}".format(type(X_list)))

        self.targets_ = y_list.columns
        self.inputs_ = X_list.columns

        for target in self.targets_:
            target_values = set(y_list[target])
            if target_values != {0, 1}:
                raise ValueError('Classifier column should contain (0, 1) values, '
                                 'got: {} in {}'.format(target_values, target), 'Please remove the column')

        self.rf.fit(X_list, y_list, sample_weight=sample_weights)

        logging.info('Top 10 features:\n %s', self.get_feature_importances().head(10))

        if sys.version_info[0] == 3:
            logging.warning('Warning! Random Forests trained with Python 3 will not be readable in Python 2! '
                            'Use Python 2.7 to generate models compatible with both versions')

        return self

    def get_feature_importances(self):
        return pd.Series(self.rf.feature_importances_, index=self.inputs_).sort_values(ascending=False)
