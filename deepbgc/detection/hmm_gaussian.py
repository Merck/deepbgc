#!/usr/bin/env python
# David Prihoda
# Gaussian HMM model for BGC domain-level prediction
# Experimental, did not get satisfactory results

import pandas as pd
from sklearn import mixture
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
from hmmlearn import hmm


class GaussianHMM(BaseEstimator, ClassifierMixin):

    def __init__(self, num_pos_means=5, num_neg_means=5, covariance_type="diag", meta=None):
        self.num_pos_means = num_pos_means
        self.num_neg_means = num_neg_means
        self.covariance_type = covariance_type
        self.meta = meta or {}

    def predict(self, X):
        # Predict posterior probability using our HMM
        logprob, posteriors = self.model_.score_samples(X)
        # BGC state probability is in second column
        return posteriors[:,1]

    def fit(self, X_list, y_list, startprob=None, transmat=None, verbose=1, debug_progress_path=None, validation_X_list=None, validation_y_list=None):
        if validation_X_list:
            print('GaussianHMM: Validation is present but has no effect yet.')
        if startprob is None:
            raise ValueError('Calculating start probability not supported yet, specify startprob explicitly')
        if transmat is None:
            raise ValueError('Calculating transition matrix not supported yet, specify transmat explicitly')

        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        pos_vectors = X[y == 1]
        neg_vectors = X[y == 0]

        if verbose:
            print('Training positive GMM on {} vectors'.format(len(pos_vectors)))
        pos_gmm = mixture.GaussianMixture(n_components=self.num_pos_means, covariance_type=self.covariance_type)
        pos_gmm.fit(pos_vectors)

        if verbose:
            print('Training negative GMM on {} vectors'.format(len(neg_vectors)))
        neg_gmm = mixture.GaussianMixture(n_components=self.num_neg_means, covariance_type=self.covariance_type)
        neg_gmm.fit(neg_vectors)

        self.model_ = GMMHMM2(n_components=2, covariance_type=self.covariance_type, verbose=bool(verbose))
        self.model_.startprob_ = startprob
        self.model_.transmat_ = transmat
        self.model_.gmms_ = np.array([neg_gmm, pos_gmm])
        return self

    def save(self, path):
        pickle.dump(self, path)
        return self

    @classmethod
    def load(cls, path):
        return pickle.load(path)

    def get_sample_emissions(self, X):
        feature_matrix = self.features.get_feature_matrix(X)
        return pd.DataFrame({
            'OUT': self.model_.gmms_[0].score_samples(feature_matrix),
            'BGC': self.model_.gmms_[1].score_samples(feature_matrix)
        })


class GMMHMM2(hmm.GMMHMM):
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 covariance_type='diag', covars_prior=1e-2,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmcw", init_params="stmcw"):
        hmm._BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm, random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

        self.covariance_type = covariance_type
        self.covars_prior = covars_prior
        self.gmms_ = []

    def _compute_log_likelihood(self, X):
        return np.array([g.score_samples(X) for g in self.gmms_]).T
