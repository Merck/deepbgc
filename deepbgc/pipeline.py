#!/usr/bin/env python
# David Prihoda
# Wrapper for a BGC detection model than handles feature transformation and loading model definitions from JSON

from deepbgc import features
import pickle
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from pprint import pprint


class DeepBGCModel(BaseEstimator, ClassifierMixin):
    """
    Wraper for a BGC detection model than handles feature transformation and loading model definitions from JSON
    """
    def __init__(self, transformer: features.ListTransformer, model, fit_params: dict, color=None, label=None):
        """

        :param transformer: ListTransformer used to transform Domain DataFrames into feature matrices
        :param model: New instance of a BGC detection model
        :param fit_params: Params to pass to the fit function of given model
        :param color: Model color stored for plotting purposes
        :param label: Model label stored for plotting purposes
        """
        self.transformer = transformer
        self.model = model
        self.fit_params = fit_params
        self.color = color
        self.label = label

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
        if validation_y is None:
            validation_y = []
        if validation_samples is None:
            validation_samples = []

        self.transformer.fit(samples, y)

        train_X_list = self.transformer.transform(samples, y)
        validation_X_list = self.transformer.transform(validation_samples, validation_y)

        merged_params = self.fit_params.copy()
        merged_params.update(extra_fit_params)
        return self.model.fit(train_X_list, y, validation_X_list=validation_X_list, validation_y_list=validation_y, **merged_params)

    def predict(self, sample):
        X_list = self.transformer.transform(sample)
        return self.model.predict(X_list)

    @classmethod
    def from_config(cls, config, meta_only=False) -> 'DeepBGCModel':
        """
        Load model configuration from a JSON config
        :param config: Path to JSON config or loaded config dict
        :param meta_only: Do not create feature transformers
        :return: Untrained pipeline based on given config
        """
        if isinstance(config, str):
            with open(config) as f:
                config = json.loads(f.read())
        elif isinstance(config, dict):
            pass
        else:
            raise AttributeError('Invalid config type "{}": {}'.format(type(config), config))

        print('Loaded model:')
        pprint(config)

        color = config.get('color', 'grey')
        label = config.get('label')
        build_params = config.get('build_params', {})
        fit_params = config.get('fit_params', {})
        input_params = config.get('input_params', {})

        # Get class from "models" module. Don't forget to import the class in models.__init__ first!
        clf_class = getattr(models, config.get('type'))

        # Create a new model instance
        model = clf_class(**build_params)

        if meta_only:
            transformer = None
        else:
            feature_params = input_params.get('features', [])
            transformer = features.ListTransformer.from_config(feature_params)

        return DeepBGCModel(transformer=transformer, model=model, fit_params=fit_params, color=color, label=label)

    def save(self, path) -> 'DeepBGCModel':
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return self

    @classmethod
    def load(cls, path) -> 'DeepBGCModel':
        with open(path, 'rb') as f:
            return pickle.load(f)

