#!/usr/bin/env python
# -*- coding: utf-8 -*-
# David Prihoda
# Wrapper for a BGC detection model than handles feature transformation and loading model definitions from JSON

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import logging
import six
from deepbgc import models, features, __version__
import pickle
import json
from sklearn.base import BaseEstimator, ClassifierMixin
import pprint
import pandas as pd
import re
import time


class SequenceModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for a sequence detection/classification model than handles feature transformation and loading model definitions from JSON
    """
    def __init__(self, transformer, model, fit_params):
        """

        :param transformer: ListTransformer used to transform Domain DataFrames into feature matrices
        :param model: New instance of a BGC detection model
        :param fit_params: Params to pass to the fit function of given model
        """
        self.transformer = transformer
        self.model = model
        self.fit_params = fit_params
        self.version = __version__
        self.timestamp = time.time()

    def fit(self, samples, y, validation_samples=None, validation_y=None, **extra_fit_params):
        """
        Train model with given list of samples, observe performance on given validation samples.
        Domain DataFrames are converted to feature matrices using the pipeline's feature transformer.
        :param samples: List of Domain DataFrames, each DataFrame contains one BGC or non-BGC sample's sequence of protein domains.
        :param y: List of output values, one value for each sequence
        :param validation_samples: List of validation samples
        :param validation_y: List of validation sample outputs
        :param extra_fit_params: Extra fitting parameters to pass to the fit function of given model
        :return: self
        """
        if validation_samples is None:
            validation_samples = []
        if validation_y is None:
            validation_y = []

        self._check_samples(samples)
        self._check_samples(validation_samples)

        self.transformer.fit(samples, y)

        train_X_list = self._safe_transform(samples, y)
        validation_X_list = self._safe_transform(validation_samples, validation_y)

        self._debug_samples(train_X_list, y)

        merged_params = self.fit_params.copy()
        merged_params.update(extra_fit_params)

        return self.model.fit(train_X_list, y, validation_X_list=validation_X_list, validation_y_list=validation_y, **merged_params)

    def _check_samples(self, samples):
        # Wrap single sample into list
        if isinstance(samples, pd.DataFrame):
            samples = [samples]
        elif not isinstance(samples, list):
            raise TypeError('Expected single sample or list of samples, got {}'.format(type(samples)))
        for sequence in samples:
            if not isinstance(sequence, pd.DataFrame):
                raise TypeError('Sample has to be a DataFrame, got ' + str(type(sequence)))

    def _debug_samples(self, X_list, y=None):
        if isinstance(X_list, pd.DataFrame):
            logging.debug('-'*80)
            logging.debug('Preview of sequence vectors X:\n%s', X_list.head(5))
            logging.debug('-'*80)
            if y is not None:
                logging.debug('Preview of response vectors y:\n%s', y.head(5))
                logging.debug('-'*80)
        elif isinstance(X_list, list) and X_list:
            logging.debug('-'*80)
            logging.debug('Preview of first sequence X:\n%s', X_list[0].head(5))
            logging.debug('-'*80)
            if y is None:
                pass
            elif isinstance(y, pd.DataFrame):
                logging.debug('Preview of response vectors y:\n%s', y.head())
                logging.debug('-'*80)
            else:
                logging.debug('Preview of first sequence y:\n%s', y[0].head())
                logging.debug('-'*80)

    def _safe_transform(self, samples, y):
        X_list = self.transformer.transform(samples)
        if isinstance(X_list, pd.DataFrame) and not X_list.empty:
            if not isinstance(y, pd.DataFrame):
                raise ValueError('In single vector sequence mode, the response needs to be a DataFrame with one row for each sample')
            if len(X_list.index) != len(y.index):
                raise ValueError('Index length does not match for sample vectors ({}) and responses ({})'.format(len(X_list.index), len(y.index)))
        return X_list

    def predict(self, samples):
        """
        Return prediction scores for each sequence in list.
        In detection, will return list of numpy arrays with prediction score for each sequence element (e.g. protein domain).
        In classification, will return a DataFrame with one row for each sequence and one column for each predicted class score.
        :param samples: List of DataFrames (sequences) or single DataFrame (sequence)
        :return: Return prediction scores for each sequence in list.
        """
        X_list = self.transformer.transform(samples)
        self._debug_samples(X_list)
        if isinstance(X_list, list):
            return [self.model.predict(X) for X in X_list]
        return self.model.predict(X_list)

    @classmethod
    def from_config(cls, config, meta_only=False, vars=None):
        """
        Load model configuration from a JSON config
        :param config: Path to JSON config or loaded config dict
        :param meta_only: Do not create feature transformers
        :param vars: Dictionary of variables to inject into JSON fields in "#{MYVAR}" format
        :return: Untrained pipeline based on given config
        """

        if isinstance(config, six.string_types):
            with open(config) as f:
                config = json.loads(f.read())
        elif isinstance(config, dict):
            pass
        else:
            raise AttributeError('Invalid config type "{}": {}'.format(type(config), config))

        config = fill_vars(config, vars)

        logging.info('Loaded model:')
        logging.info(pprint.pformat(config, indent=4))

        build_params = config.get('build_params', {})
        fit_params = config.get('fit_params', {})
        input_params = config.get('input_params', {})
        sequence_as_vector = input_params.get('sequence_as_vector', False)

        # Get class from "models" module. Don't forget to import the class in models.__init__ first!
        clf_class = getattr(models, config.get('type'))

        # Create a new model instance
        model = clf_class(**build_params)

        if meta_only:
            transformer = None
        else:
            feature_params = input_params.get('features', [])
            transformer = features.ListTransformer.from_config(feature_params, sequence_as_vector=sequence_as_vector)

        return SequenceModelWrapper(transformer=transformer, model=model, fit_params=fit_params)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=2)
        return self

    @classmethod
    def load(cls, path):
        logging.info('Loading model from: {}'.format(path))
        try:
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
            except UnicodeDecodeError:
                with open(path, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
        except ImportError as e:
            if 'hmmlearn' in str(e):
                from deepbgc.models.hmm import get_hmmlearn_import_error
                raise get_hmmlearn_import_error()
            raise e
        except Exception as e:
            raise ValueError("Error unpickling model from path '{}'".format(path), e)

        if not isinstance(model, cls):
            raise TypeError("Provided model is not a SequenceModelWrapper: '{}' is a {}".format(path, type(model)))
        return model


VAR_PATTERN = re.compile("(#{([a-zA-Z_0-9]+)})")
def fill_vars(d, vars):
    if vars is None:
        vars = {}
    if isinstance(d, dict):
        return {k: fill_vars(v, vars) for k, v in d.items()}
    elif isinstance(d, list):
        return [fill_vars(v, vars) for v in d]
    elif isinstance(d, six.string_types):
        return VAR_PATTERN.sub(lambda match: _get_matched_var(match, vars), d)
    return d


def _get_matched_var(match, vars):
    name = match.group(2)
    if name not in vars:
        raise ValueError("Missing config variable {}, specify it using --config {} VALUE".format(name, name))
    return vars[name]
