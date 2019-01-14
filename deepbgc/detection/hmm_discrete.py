#!/usr/bin/env python
# David Prihoda
# HMM models for BGC domain-level prediction
# Emission probability can be calculated from positive and negative training samples.
# Starting and transition probability have to be provided.

import numpy as np
from hmmlearn import hmm
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import os


class HMM(BaseEstimator, ClassifierMixin):
    """
    HMM model parent class providing Sklearn mixins and saving/loading functionality
    """
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return self

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class DiscreteHMM(HMM):

    def get_sample_vector(self, X):
        """
        Turn pfam IDs into integers based on our vocabulary
        :param X: DataFrame of domains with pfam_id column
        :return: numpy array of numbers representing given words in our vocabulary
        """
        return np.array([self.vocabulary_.get(o, -1) for o in X['pfam_id']])

    def predict(self, X: pd.DataFrame):
        """
        Get BGC prediction score for a Domain DataFrame
        :param X: DataFrame with pfam domains
        :return: numpy array of BGC prediction scores for each domain in X
        """
        word_vector = self.get_sample_vector(X)
        # Predict posterior probability using our HMM
        logprob, posteriors = self.model_.score_samples(word_vector.reshape(-1, 1))
        # BGC state probability is in second column
        return posteriors[:,1]

    def _get_pfam_counts(self, X, y):
        """
        Get number of occurences of each pfam ID in negative (non-BGC) and positive (BGC) states
        :param X: Domain DataFrame with pfam_id column
        :param y: Series of states for each domain (0 = non-BGC, 1 = BGC)
        :return: DataFrame with number of positive and negative occurences (pos and neg columns) of each pfam_id (index).
        """
        counts = X[['pfam_id']].drop_duplicates().set_index('pfam_id')
        unique_y = set(y)
        if unique_y != {0, 1}:
            raise AttributeError('Invalid target values, expected {0, 1} got '+str(unique_y))
        counts['pos'] = X[y == 1]['pfam_id'].value_counts()
        counts['neg'] = X[y == 0]['pfam_id'].value_counts()
        return counts.fillna(0)

    def _construct_model(self, startprob, transmat, emissionprob, vocabulary):
        """
        Create internal HMM model with given matrices and store it to self.model_
        :param startprob: Starting probability [negative_starting_prob, positive_starting_prob]
        :param transmat: Transition matrix (An array where the [i][j]-th element corresponds to the posterior probability of transitioning between the i-th to j-th)
        :param emissionprob: Emission probability [[neg_pfam1, neg_pfam2, ...], [pos_pfam1, pos_pfam2, ...]] with pfam IDs indexed by their vocabulary index numbers
        :param vocabulary: Vocabulary dictionary with {pfam_id: index_number_in_emission}
        :return: self
        """
        self.model_ = hmm.MultinomialHMM(n_components=2)
        if isinstance(startprob, list):
            startprob = np.array(startprob)
        if isinstance(transmat, list):
            transmat = np.array(transmat)
        self.model_.startprob_ = startprob
        self.model_.transmat_ = transmat
        self.model_.emissionprob_ = emissionprob
        self.vocabulary_ = vocabulary
        return self

    def fit(self, X_list, y_list, sample_weights=None, startprob=None, transmat=None, verbose=0,
            default_emission_count=0.01, debug_progress_path=None, validation_X_list=None, validation_y_list=None):
        """
        Create and train internal HMM model based on list of positive and negative samples.
        Emission probability will be calculated from samples. Starting and transition probability have to be provided.

        :param X_list: List of samples (Domain DataFrames)
        :param y_list: List of sample states (0 or 1), one value for each sample (DataFrame)
        :param sample_weights: List of sample weights, marking their contribution to the emission probability. If not provided, will be set to 1 for all samples.
        :param startprob: Starting probability [negative_starting_prob, positive_starting_prob]
        :param transmat: Transition matrix (An array where the [i][j]-th element corresponds to the posterior probability of transitioning between the i-th to j-th)
        :param verbose: Verbosity (0 = no output, 1 = plot top pfams for positive and negative states)
        :param default_emission_count: Emission value for the other state for pfams that appear only in the positive / negative state
        :param debug_progress_path: Not used in HMM models.
        :param validation_X_list: List of validation samples, not used in HMM models.
        :param validation_y_list: List of validation states, not used in HMM models.
        :return: self
        """
        if validation_X_list:
            print('DiscreteHMM: Validation is present but has no effect yet.')
        if startprob is None:
            raise ValueError('Calculating start probability not supported yet, specify startprob explicitly')
        if transmat is None:
            raise ValueError('Calculating transition matrix not supported yet, specify transmat explicitly')

        if sample_weights is not None:
            zipped = enumerate(zip(X_list, y_list, sample_weights))
            weighted_counts = [self._get_pfam_counts(X, y) * weight for i, (X, y, weight) in zipped]
            all_counts = pd.concat(weighted_counts).reset_index().groupby('pfam_id').sum().sort_index()
        else:
            X: pd.DataFrame = pd.concat(X_list)
            y: pd.DataFrame = pd.concat(y_list)
            all_counts = self._get_pfam_counts(X, y).sort_index()

        if verbose:
            print('Top positive:')
            print(all_counts.sort_values(by='pos', ascending=False).head(3))
            print('Top negative:')
            print(all_counts.sort_values(by='neg', ascending=False).head(3))

        # For a pfam_id that appears only in the positive / negative state, set the default emission count instead of 0
        all_counts.replace(0, default_emission_count, inplace=True)

        # Vocabulary stores map of pfam_id -> index in emission vector
        vocabulary = {pfam_id: i for i, pfam_id in enumerate(all_counts.index)}

        emissions = all_counts[['neg', 'pos']].values
        # Divide each state's emission counts by the total number of observations to get emission frequency
        emissions /= emissions.sum(axis=0)
        # Add default emissions for unseen pfam_ids to the end (will be indexed by -1)
        emissions = np.concatenate([emissions, np.array([[0.5, 0.5]])])

        self._construct_model(startprob, transmat, emissions.T, vocabulary)
        return self

    def get_sample_emissions(self, sample):
        word_index = self.get_sample_vector(sample)
        return pd.DataFrame({
            'OUT': [None if x == -1 else self.model_.emissionprob_[0][x] for x in word_index],
            'BGC': [None if x == -1 else self.model_.emissionprob_[1][x] for x in word_index]
        })


class GeneBorderHMM(HMM):
    """
    HMM that only changes its state at gene borders.
    Implemented by turning each input symbol (pfam ID) into a tuple of (pfam ID, is_at_gene_end)
    and each negative and positive state into four states with tuples (positive/negative, is_at_gene_end)

    Emissions at gene ends have 0 emission probability in states that are not at gene ends and vice versa.
    Transitions can only happen from states where is_at_gene_end = True, which means probability is set to 0 for all other transitions.
    """
    def _convert_startprob(self, startprob):
        if startprob is None:
            return
        # Start probability
        start_out = startprob[0]
        start_bgc = startprob[1]
        return np.array([start_out / 2, start_out / 2, start_bgc / 2, start_bgc / 2])
        #print('Converted to four state start probability:')
        #print(self.model.startprob_)

    def _convert_transmat(self, transmat, X_list, verbose=0):
        if transmat is None:
            return

        num_gene_end = sum([sum(get_sample_gene_ends(X['protein_id'])) for X in X_list])
        num_total = sum([len(X) for X in X_list])
        frac_in_gene_end = num_gene_end / num_total
        if verbose:
            print('Gene end: {} ({}/{})'.format(frac_in_gene_end, num_gene_end, num_total))

        # Transition probability
        out2bgc = transmat[0][1] * frac_in_gene_end
        out2out = 1 - out2bgc
        bgc2out = transmat[1][0] * frac_in_gene_end
        bgc2bgc = 1 - bgc2out

        converted = np.array([
            [0.5, 0.5, 0, 0],
            [out2out / 2, out2out / 2, out2bgc / 2, out2bgc / 2],
            [0, 0, 0.5, 0.5],
            [bgc2out / 2, bgc2out / 2, bgc2bgc / 2, bgc2bgc / 2]
        ])
        if verbose:
            print('Converted to four state transitions:')
            print(converted)
        return converted

    def _convert_emission(self, old_emissionprob, old_vocabulary):
        # Emissions
        num_words = len(old_vocabulary)
        out_emissions = old_emissionprob[0][:-1]
        bgc_emissions = old_emissionprob[1][:-1]
        zero_emissions = np.zeros(num_words)
        default_emission = old_emissionprob[0][-1]

        emissionprob = np.zeros((4, num_words * 2 + 2))
        emissionprob[0] = np.concatenate([out_emissions, zero_emissions, [default_emission, 0]])
        emissionprob[1] = np.concatenate([zero_emissions, out_emissions, [0, default_emission]])
        emissionprob[2] = np.concatenate([bgc_emissions, zero_emissions, [default_emission, 0]])
        emissionprob[3] = np.concatenate([zero_emissions, bgc_emissions, [0, default_emission]])

        # Vocabulary
        vocabulary = {}
        for pfam_id, word_index in old_vocabulary.items():
            vocabulary[(pfam_id, False)] = word_index
            vocabulary[(pfam_id, True)] = word_index + num_words

        return emissionprob,  vocabulary

    def _get_word_index(self, pfam_id, is_gene_end):
        default_index = -1 if is_gene_end else -2
        return self.vocabulary_.get((pfam_id, is_gene_end), default_index)

    def get_sample_vector(self, X):
        is_gene_end = get_sample_gene_ends(X['protein_id'])
        if not any(is_gene_end):
            print('Warning: no gene end predicted: '+str(X.head(1)))
        return np.array([self._get_word_index(x, is_gene_end[i]) for i, x in enumerate(X['pfam_id'].values)])

    def predict(self, X):
        sample_vector = self.get_sample_vector(X)
        prev_level = np.geterr()['divide']
        np.seterr(divide='ignore')
        logprob, posteriors = self.model_.score_samples(sample_vector.reshape(-1, 1))
        np.seterr(divide=prev_level)
        # final prediction is maximum of the probability of the last two states
        prediction = posteriors[:,2:]
        return np.max(prediction, axis=1)

    def fit(self, X_list, y_list, startprob=None, transmat=None, verbose=1, debug_progress_path=None, validation_X_list=None, validation_y_list=None):
        if validation_X_list:
            print('GeneBorderHMM: Validation is present but has no effect yet.')
        if verbose:
            print('Training two state model...')

        two_state_model = DiscreteHMM()
        two_state_model.fit(X_list, y_list, startprob=startprob, transmat=transmat, verbose=verbose)

        emission, self.vocabulary_ = self._convert_emission(two_state_model.model_.emissionprob_, two_state_model.vocabulary_)

        self.model_ = hmm.MultinomialHMM(n_components=4)
        self.model_.startprob_ = self._convert_startprob(startprob)
        self.model_.transmat_ = self._convert_transmat(transmat, X_list)
        self.model_.emissionprob_ = emission
        return self

    def get_sample_emissions(self, X):
        sample_vector = self.get_sample_vector(X)
        return pd.DataFrame({
            'OUT_IN_GENE': [None if x < 0 else self.model_.emissionprob_[0][x] for x in sample_vector],
            'OUT_GENE_END': [None if x < 0 else self.model_.emissionprob_[1][x] for x in sample_vector],
            'BGC_IN_GENE': [None if x < 0 else self.model_.emissionprob_[2][x] for x in sample_vector],
            'BGC_GENE_END': [None if x < 0 else self.model_.emissionprob_[3][x] for x in sample_vector]
        })


class ClusterFinderHMM(DiscreteHMM):
    """
    Wrapper that loads the ClusterFinder trained model from the pickled starting, transition and emission matrices.
    """
    def fit(self, X_unused, y_unused, param_dir=None, **kwargs):

        with open(os.path.join(param_dir, 'NewTS_all_B_index.pkl'), 'rb') as pfile:
            cf_vocabulary = pickle.load(pfile)

        # Start probability
        with open(os.path.join(param_dir, 'SP_arr.pkl'), 'rb') as pfile:
            cf_start = pickle.load(pfile, encoding='latin1')

        # Transition probability between states
        with open(os.path.join(param_dir, 'TP_arr_A.pkl'), 'rb') as pfile:
            cf_transition = pickle.load(pfile, encoding='latin1')

        # Emission probability for each state and pfam
        with open(os.path.join(param_dir, 'NewTS_all_B_reduced_6filter.pkl'), 'rb') as pfile:
            cf_emission = pickle.load(pfile, encoding='latin1')

        # Add default emission to the end of the emission matrix
        # Default emission is used when the observed sequence contains words that didn't appear in our vocabulary
        # The value actually does not matter as long as it's the same for both states
        cf_default_emission = 1.6026668376177961e-07
        cf_default_emissions = np.array([[cf_default_emission], [cf_default_emission]])
        cf_emission = np.append(cf_emission, cf_default_emissions, axis=1)
        print('Default emission', cf_default_emission)

        # Create HMM with given parameters
        # States are flipped to use more intuitive NONBGC=0, BGC=1
        startprob = np.array([cf_start[1], cf_start[0]])
        transmat = np.array([[cf_transition[1][1], cf_transition[1][0]],
                             [cf_transition[0][1], cf_transition[0][0]]])
        emissionprob = np.array([cf_emission[1], cf_emission[0]])
        print('Start probability (0=NONBGC, 1=BGC):\n', startprob)
        print('Transition probability (0=NONBGC, 1=BGC):\n', transmat)
        print('Emission probability (0=NONBGC, 1=BGC):\n', emissionprob)

        self._construct_model(startprob=startprob, transmat=transmat, emissionprob=emissionprob, vocabulary=cf_vocabulary)
        return self


def get_sample_gene_ends(gene_ids):
    """
    For list of Gene IDs, return list of boolean values that mark whether the next gene is different (or we are at end of sequence)
    :param gene_ids: List of gene IDs
    :return: list of boolean values that mark whether the next gene is different (or we are at end of sequence)
    """
    gene_ends = list(gene_ids[:-1].values != gene_ids[1:].values) + [True]
    return np.array(gene_ends).astype(np.uint8)

