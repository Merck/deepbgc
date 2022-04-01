#!/usr/bin/env python
# David Prihoda
# Feature transformers that turn Domain DataFrames into protein feature vector matrices
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import sys


class ListTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper for other transformers, will transform each DataFrame in a list by each transformer and merge the results.
    """
    def __init__(self, transformers, sequence_as_vector=False):
        self.transformers = transformers
        self.sequence_as_vector = sequence_as_vector

    def transform(self, samples):
        if samples is None:
            return None
        if not self.transformers:
            return samples
        if isinstance(samples, pd.DataFrame):
            return self._transform_sequence(samples)
        elif not isinstance(samples, list):
            raise AttributeError('Sequences have to be a list, got ' + str(type(samples)))
        X_list = [self._transform_sequence(sequence) for sequence in samples]
        if self.sequence_as_vector:
            # Merge list of Series into a single DataFrame indexed by name of each Series (sequence ID)
            return pd.DataFrame(X_list)
        return X_list

    def _transform_sequence(self, sequence):
        if self.sequence_as_vector:
            # Output of each transformer should be a Series, merge into one long Series
            return pd.concat([t.transform(sequence) for t in self.transformers], sort=False)
        # Output of each transformer should be a matrix, merge all columns into one wide matrix
        return pd.concat([t.transform(sequence) for t in self.transformers], sort=False, axis=1)

    def fit(self, X_list, y_list=None):
        if X_list is None:
            return self
        if isinstance(X_list, pd.Series):
            X_list = list(X_list.values)
        if not isinstance(X_list, list):
            raise AttributeError('X_list has to be a list, got {}'.format(type(X_list)))
        if len(X_list):
            X_merged = pd.concat(X_list, sort=False)
            y_merged = y_list if isinstance(y_list, pd.DataFrame) else pd.concat(y_list)
            for t in self.transformers:
                t.fit(X_merged, y_merged)
        return self

    @classmethod
    def from_config(cls, transformer_configs, sequence_as_vector=False):
        transformers = []
        for params in transformer_configs:
            classname = params.get('type')
            transformer = getattr(sys.modules[__name__], classname)
            trans_args = {k: v for k, v in params.items() if k != 'type'}
            transformers.append(transformer(**trans_args))
        return ListTransformer(transformers, sequence_as_vector=sequence_as_vector)


class Pfam2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Get pfam2vec matrix for a Domain DataFrame
    """
    def __init__(self, vector_path):
        self.vector_path = vector_path
        if vector_path.endswith('.csv'):
            self.vectors = pd.read_csv(vector_path).set_index('pfam_id')
        elif vector_path.endswith('.tsv'):
            self.vectors = pd.read_csv(vector_path, sep='\t').set_index('pfam_id')
        elif vector_path.endswith('.pkl') or vector_path.endswith('.pickle'):
            self.vectors = pd.read_pickle(vector_path)
        elif vector_path.endswith('.bin'):
            import word2vec
            model = word2vec.load(vector_path, kind='bin')
            self.vectors = pd.DataFrame(model.vectors, index=model.vocab)
        else:
            raise ValueError("File type {} not supported for Pfam2Vec, use .csv, .pkl, .pickle or .bin".format(vector_path))

        vectors_min = self.vectors.min()
        too_low_idx = vectors_min < -1
        cols_too_low = list(self.vectors.columns[too_low_idx])
        if cols_too_low:
            raise ValueError('Pfam2vec vectors should be >= -1, got {} in {}'.format(list(vectors_min[too_low_idx]), cols_too_low))

        vectors_max = self.vectors.max()
        too_high_idx = vectors_max > 1
        cols_too_high = list(self.vectors.columns[too_high_idx])
        if cols_too_high:
            raise ValueError('Pfam2vec vectors should be <= 1, got {} in {}'.format(list(vectors_max[too_high_idx]), cols_too_high))

    def transform(self, X):
        # Turn each pfam ID into a vector

        df = self.vectors.reindex(index=X['pfam_id'], fill_value=0)
        df.index = X.index
        return df

    def fit(self, X, y=None):
        return self


class RandomVecTransformer(BaseEstimator, TransformerMixin):
    """
    Get random vector matrix for a Domain DataFrame. Each unique pfam_id will have the same random vector throughout the sequence.
    """

    def __init__(self, dimensions=100):
        self.dimensions = dimensions
        self.zero_vector = np.zeros(self.dimensions)
        self.vectors = {}
        self.random = np.random.RandomState(seed=0)

    def transform(self, X):
        # Turn each pfam ID into a vector
        return np.array([self.vectors.get(pfam_id, self.zero_vector) for pfam_id in X['pfam_id']])
        #print(X.iloc[0]['pfam_id'], vectors[0])
        #return vectors

    def fit(self, X, y=None):
        for pfam_id in X['pfam_id'].unique():
            if pfam_id not in self.vectors:
                self.vectors[pfam_id] = self.random.rand(self.dimensions)
        return self


class EmissionProbabilityTransformer(BaseEstimator, TransformerMixin):
    """
    Get emission probability feature column for given Domain DataFrame. Based on HMM emissions.
    """
    def __init__(self):
        self.emissions = None

    def fit(self, X, y=None):
        unique_y = set(y)
        if unique_y != {0, 1}:
            raise AttributeError('Invalid target values, expected {0, 1} got ' + str(unique_y))
        counts = pd.DataFrame(data={}, index=X['pfam_id'].unique())
        counts['neg'] = X[y == 0]['pfam_id'].value_counts()
        counts['pos'] = X[y == 1]['pfam_id'].value_counts()
        counts = counts.fillna(0)
        # Divide each state's emission counts by the total number of observations to get emission frequency
        self.emissions = counts / counts.sum(axis=0)
        return self

    def transform(self, X):
        # Turn each pfam ID into a vector
        vectors = self.emissions.reindex(index=X['pfam_id'], fill_value=0)
        return vectors


class PositiveProbabilityTransformer(BaseEstimator, TransformerMixin):
    """
    Get "positive probability" feature columns for given Domain DataFrame.
    Each pfam_id will get two columns: Positive probability and Total probability
    Positive probability = probability of being in positive state while seeing given pfam,
      which is equivalent to number of occurences of given pfam in the positive state divided by number of occurences in all states
    Total probability = probability of seeing given pfam in general,
      which is equivalent to number of occurences divided by total length of input sequence
    """
    def __init__(self):
        self.probs = None

    def fit(self, X, y=None):
        vals = pd.DataFrame({'pfam_id': X['pfam_id'], 'y': y})
        negweight = sum(y) / sum(y == 0)
        total_num_weighted = sum(y) + sum(y == 0) * negweight
        probs = {}
        for pfam_id, pfam_y in vals.groupby('pfam_id')['y']:
            num_pos = sum(pfam_y)
            num_neg = sum(pfam_y == 0)
            num_weighted = num_pos + num_neg * negweight
            prob = num_pos / num_weighted
            prob = (prob - 0.5) * 2
            pfam_frac = num_weighted / total_num_weighted
            probs[pfam_id] = [prob, pfam_frac]
        self.probs = pd.DataFrame(probs).transpose()
        return self

    def transform(self, X):
        # Turn each pfam ID into a vector
        vectors = self.probs.reindex(index=X['pfam_id'], fill_value=0)
        return vectors


class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    Create a binary one-hot-encoding vector from Domain CSV files.
    If sequence_as_vector = True, will produce a set-encoding vector of the whole sequence. Otherwise will create one vector per each Pfam domain.
    """
    def __init__(self, column='pfam_id', sequence_as_vector=True):
        self.unique_values = []
        self.column = column
        self.sequence_as_vector = sequence_as_vector

    def transform(self, X):
        if X.empty:
            X = pd.DataFrame({self.column: []})
        if self.column not in X.columns:
            raise ValueError('Unexpected pfam dataframe, column "{}" not found in: {}'.format(self.column, X.columns))
        # Turn each pfam ID into a vector
        values = pd.get_dummies(X[self.column]).reindex(columns=self.unique_values, fill_value=0)
        if self.sequence_as_vector:
            return pd.Series(values.sum().astype(np.bool).astype(np.int))
        return values

    def fit(self, X, y=None):
        self.unique_values = np.union1d(self.unique_values, X[self.column])
        return self


class ProteinBorderTransformer(BaseEstimator, TransformerMixin):
    """
    Get gene beginning / gene end binary flags from Domain CSV files.
    """

    def __init__(self, field='protein_id'):
        self.field = field

    def transform(self, X):
        # current != next (exclude last element because we don't have a next value)
        borders = list(X[self.field][:-1].values != X[self.field][1:].values)
        return pd.DataFrame({
            'protein_start': np.array([True] + borders),
            'protein_end': np.array(borders + [True])
        }, index=X.index)[['protein_start','protein_end']].astype(np.uint8)

    def fit(self, X, y=None):
        return self


class GeneDistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Returns vector specifying nucleotide distance from the end of previous gene to the start of current gene for first domain in each protein.
    Distance between domains in same protein is 0. Distance in first domain of given sample is equal to its gene_start.
    """

    def __init__(self, norm_distance):
        logging.warning('Warning! Do not use the distance transformer for merged samples - the distance would be invalid on sample borders')
        self.norm_distance = norm_distance

    def transform(self, X):
        gene_starts = X['gene_start'].values
        gene_ends = X['gene_end'].values
        previous_gene_ends = np.concatenate([[0], gene_ends[:-1]])
        distances = (gene_starts - previous_gene_ends) / self.norm_distance
        # replace negative values with zeros
        distances *= distances >= 0
        return distances.astype(np.float32).reshape(-1, 1)

    def fit(self, X, y=None):
        return self

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    """
    Select given columns of input DataFrame
    """
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        return X.select(self.columns, axis=1).values

    def fit(self, X, y=None):
        return self