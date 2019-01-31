#!/usr/bin/env python
# David Prihoda
# Generic LSTM wrapper used for the DeepBGC model

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasRNN(BaseEstimator, ClassifierMixin):
    """
    Generic LSTM wrapper used for the DeepBGC model
    """
    def __init__(self, trained_model=None, batch_size=1, hidden_size=128, loss='binary_crossentropy', stateful=True,
                 activation='sigmoid', return_sequences=True):
        from keras.models import Sequential
        if trained_model is not None:
            self.model = trained_model  # type: Sequential
            # Set the attributes from the model object to be able to clone and cross-validate a loaded model
            self.batch_size = trained_model.layers[0].batch_input_shape[0]
            self.hidden_size = trained_model.layers[0].layer.units
            self.stateful = trained_model.layers[0].layer.stateful
            self.loss = trained_model.loss
            self.activation = trained_model.layers[-1].layer.activation
            self.return_sequences = trained_model.layers[0].layer.return_sequences
        else:
            self.model = None  # type: Sequential
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.loss = loss
            self.stateful = stateful
            self.activation = activation
            self.return_sequences = return_sequences

    def _build_model(self, input_size, stacked_sizes=None, fully_connected_sizes=None, optimizer_name=None, learning_rate=None, decay=None, gpus=0, custom_batch_size=None):
        """
        Build Keras Sequential model architecture with given parameters
        :param input_size: Dimensionality of input vector (number of features)
        :param stacked_sizes: Add given number of additional Bi-LSTM layers after first Bi-LSTM layer, provided as list of sizes
        :param fully_connected_sizes: Add a given number of additional fully connected layers after the Bi-LSTM layers, provided as list of sizes
        :param optimizer_name: Name of Keras optimizer, default 'adam'
        :param learning_rate: Keras learning rate
        :param decay: Optimizer decay
        :param gpus: Number of gpus to train on (Not implemented)
        :param custom_batch_size: Use different batch size than self.batch_size
        :return: Keras Sequential model
        """
        from keras.layers.core import Dense
        from keras.layers.recurrent import LSTM
        from keras.layers.wrappers import TimeDistributed, Bidirectional
        from keras.models import Sequential
        from keras import optimizers
        if stacked_sizes is None:
            stacked_sizes = []
        if fully_connected_sizes is None:
            fully_connected_sizes = []

        model = Sequential()

        model.add(Bidirectional(
            layer=LSTM(
                units=self.hidden_size,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                stateful=self.stateful
            ),
            batch_input_shape=(custom_batch_size or self.batch_size, None, input_size)
        ))

        for size in stacked_sizes:
            model.add(Bidirectional(layer=LSTM(units=size, return_sequences=True, stateful=self.stateful)))

        for size in fully_connected_sizes:
            model.add(TimeDistributed(Dense(size, activation='sigmoid')))

        model.add(TimeDistributed(Dense(1, activation='sigmoid')))

        if gpus > 1:
            raise NotImplementedError("Multi GPU model not implemented due to input size mismatch.")
            #model = multi_gpu_model(model, gpus=gpus)

        if optimizer_name is None:
            optimizer_name = "adam"

        optimizer_args = {}
        if learning_rate is not None:
            optimizer_args['lr'] = learning_rate
        if decay is not None:
            optimizer_args['decay'] = decay

        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(**optimizer_args)
        elif optimizer_args:
            raise ValueError('Optimizer {} not implemented for custom params yet'.format(optimizer_name))
        else:
            optimizer = optimizer_name

        print('Using optimizer', optimizer_name, optimizer_args)
        model.compile(loss=self.loss, optimizer=optimizer, sample_weight_mode='temporal', metrics=["accuracy", precision, recall, auc_roc])
        return model

    def fit(self, X_list, y_list, timesteps=128, validation_size=0.33, num_epochs=10, verbose=1,
            debug_progress_path=None, fully_connected_sizes=None,
            shuffle=True, gpus=0, stacked_sizes=None, early_stop_mode=None, early_stop_monitor=None, early_stop_min_delta=0.005, early_stop_patience=10,
            positive_weight=None, weighted=False, optimizer=None, learning_rate=None, decay=None,
            validation_X_list=None, validation_y_list=None):
        """
        Train Keras Sequential model using provided list of positive / negative samples.
        Training is done in given number of epochs with additional stopping criteria.
        In each epoch, we go over all samples in X_list, which are shuffled randomly and merged together into artificial genomes.

        :param X_list: List of DataFrames (samples) where each DataFrame contains protein domains represented by numeric vectors
        :param y_list: List of output values, one value for each sample where 0 = negative sample (non-BGC), 1 = positive sample (BGC)
        :param timesteps: Number of timesteps (protein domains) in one batch
        :param validation_size: Fraction of samples to use for testing
        :param num_epochs: Number of epochs. If early stopping is defined, this serves as a limit of maximum number of epochs.
        :param verbose: Verbosity (0 = silent, 1 = verbose, 2 = very verbose)
        :param debug_progress_path: Log Tensorboard information in given folder
        :param fully_connected_sizes: Add a given number of additional fully connected layers after the Bi-LSTM layers, provided as list of sizes
        :param shuffle: Whether to shuffle samples within each epoch. If not used, make sure that positive and negative samples are already shuffled in the list.
        :param gpus: Number of gpus to use (not implemented!)
        :param stacked_sizes: Add given number of additional Bi-LSTM layers after first Bi-LSTM layer, provided as list of sizes
        :param early_stop_mode: Keras early stopping mode (use max for increasing metrics like AUC ROC, use min for decreasing metrics like Loss)
        :param early_stop_monitor: Metric to observe for early stopping (e.g. val_auc_roc)
        :param early_stop_min_delta: Minimum change to observed metric needed to continue training
        :param early_stop_patience: Number of epochs to get maximum value of the observed metric from, if that value does not improve over the previous maximum, stop training
        :param positive_weight: Weight of positive samples (single number). Can be used to counter imbalance in training data.
        :param weighted: Calculate positive weight automatically as num negatives / num positive samples in input training data (y_list).
        :param optimizer: Name of Keras optimizer, default 'adam'.
        :param learning_rate: Keras learning rate
        :param decay: Keras optimizer decay.
        :param validation_X_list: List of DataFrames (samples) used to observe validation performance
        :param validation_y_list: List of output values for validation samples, one value for each sample where 0 = negative sample (non-BGC), 1 = positive sample (BGC)
        :return: self
        """

        import keras

        if not isinstance(X_list, list):
            raise AttributeError('Expected X_list to be list, got ' + str(type(X_list)))

        if not isinstance(y_list, list):
            raise AttributeError('Expected y_list to be list, got ' + str(type(X_list)))

        if weighted:
            if positive_weight:
                raise ValueError('Positive weight cannot be specified together with weighted=true.')
            num_neg = _count_samples(y_list, 0)
            num_pos = _count_samples(y_list, 1)
            positive_weight = num_neg / num_pos
            print('Negative: {}, Positive: {}'.format(num_neg, num_pos))
            print('Weighing positives based on ratio, weight:', positive_weight)

        input_size = X_list[0].shape[1]

        train_model = self._build_model(input_size, stacked_sizes, fully_connected_sizes=fully_connected_sizes, optimizer_name=optimizer, learning_rate=learning_rate, decay=decay, gpus=gpus)
        self.model = self._build_model(input_size, stacked_sizes, fully_connected_sizes=fully_connected_sizes, optimizer_name=optimizer, learning_rate=learning_rate, decay=decay, gpus=gpus, custom_batch_size=1)

        X_train, y_train = X_list, y_list
        validation_data, validation_num_batches = None, None

        if validation_X_list:
            if positive_weight:
                print('Warning: Not using positive_weight "{}" on external validation set!'.format(positive_weight))
            if validation_size:
                print('Warning: LSTM validation size {} specified but ignored, '
                      'because external validation set is also present.'.format(validation_size))

            print('Validating on external validation set of {} samples'.format(len(validation_X_list)))
            validation_data = _repeat_to_fill_batch_size(validation_X_list, validation_y_list, self.batch_size, input_size)
            validation_num_batches = None
        elif validation_size:
            print('Validating on {:.1f}% of input set'.format(validation_size*100))
            X_train, X_validation, y_train, y_validation = train_test_split(X_list, y_list, test_size=validation_size)

            get_validation_gen, validation_num_batches = _build_generator(
                X_validation,
                y_validation,
                batch_size=self.batch_size,
                timesteps=timesteps,
                input_size=input_size,
                shuffle=shuffle,
                positive_weight=positive_weight
            )
            validation_data = get_validation_gen()

        get_train_gen, train_num_batches = _build_generator(
            X_train,
            y_train,
            batch_size=self.batch_size,
            timesteps=timesteps,
            input_size=input_size,
            shuffle=shuffle,
            positive_weight=positive_weight,
        )
        train_gen = get_train_gen()


        callbacks = []
        if debug_progress_path:
            tb = keras.callbacks.TensorBoard(log_dir=debug_progress_path, histogram_freq=0, batch_size=self.batch_size,
                                             write_graph=True,
                                             write_grads=False, write_images=False,
                                             embeddings_layer_names=None, embeddings_metadata=None)
            callbacks.append(tb)

        if early_stop_monitor:
            if not early_stop_mode:
                raise ValueError('Keras early_stop_mode has to be specified (min, max, auto) to enable early_stop_monitor.')

            callbacks.append(keras.callbacks.EarlyStopping(
                min_delta=early_stop_min_delta,
                monitor=early_stop_monitor,
                patience=early_stop_patience,
                mode=early_stop_mode,
                verbose=1
            ))

        with _get_device(gpus):
            history = train_model.fit_generator(
                generator=train_gen,
                steps_per_epoch=train_num_batches,
                shuffle=False,
                epochs=num_epochs,
                validation_data=validation_data,
                validation_steps=validation_num_batches,
                callbacks=callbacks,
                verbose=verbose
            )

        trained_weights = train_model.get_weights()
        self.model.set_weights(trained_weights)

        return history

    def predict(self, X):
        """
        Predict given sample DataFrame/numpy matrix of numeric protein vectors
        :param X: DataFrame/numpy matrix of protein vectors
        :return: BGC prediction score for each protein vector
        """
        if len(X.shape) != 2:
            raise AttributeError('Can only be called on a single 2-dimensional feature matrix.')

        if self.model is None:
            raise AttributeError('Cannot predict using untrained model.')

        batch_matrix = X.reshape(1, X.shape[0], X.shape[1])

        # TODO do we need to reset here?
        self.model.reset_states()
        probs = self.model.predict(batch_matrix, batch_size=1)
        return probs[0,:,0]

    def save(self, path):
        if self.model is None:
            raise AttributeError('Cannot save untrained model.')
        self.model.save(path)
        return self

    @classmethod
    def load(cls, path):
        import keras
        model = keras.models.load_model(path, custom_objects={'precision': precision, 'recall': recall, 'auc_roc': auc_roc})
        return KerasRNN(trained_model=model)

    def __getstate__(self):
        """
        Get representation of object that can be pickled
        :return: objects to be pickled
        """
        attrs = self.__dict__.copy()
        del attrs['model']

        if self.model is None:
            return attrs, None, None
        return attrs, self.model.to_json(), self.model.get_weights()

    def __setstate__(self, state):
        from keras.models import Sequential, model_from_json
        """
        Load object from pickled representation
        :param state: attributes of model generated by __getstate__
        """
        attrs, architecture, weights = state

        self.__dict__.update(attrs)

        if architecture is None:
            self.model = None
        else:
            self.model = model_from_json(architecture)  # type: Sequential
            self.model.set_weights(weights)

def rotate(l, n):
    m = n % len(l)
    return l[m:] + l[:m]

def _noop():
    return None

def _yield_single_pair(a, b):
    yield a, b

def _repeat_to_fill_batch_size(X_list, y_list, batch_size, input_size):
    """
    Fill matrix of batch_size rows with samples from X_list in a way that all samples are (approximately) evenly present.
    Create batch_size rows, each row as long as the longest sample in X_list (max_len).
    For row on index i, include concatenated sequence of X_list starting from sample i (sequence is trimmed to max_len).
    :param X_list: list of samples
    :param y_list: list of sample responses
    :param batch_size: how many rows to create
    :param input_size: number of columns in sample
    :return: Filled matrix of batch_size rows with samples from X_list in a way that all samples are (approximately) evenly present.
    """
    if len(X_list) > batch_size:
        raise AttributeError('Cannot repeat more samples than batch_size.')

    max_len = max([X.shape[0] for X in X_list])

    fill_shape = (batch_size, max_len, input_size)
    fill_num_values = fill_shape[0] * fill_shape[1] * fill_shape[2]
    print('Filling to batch size shape {} ({}M values)...'.format(fill_shape, int(fill_num_values / 1000000)))

    X_filled = np.zeros(shape=fill_shape)
    y_filled = np.zeros(shape=(fill_shape[0], fill_shape[1], 1))

    for i in range(0, batch_size):
        X_filled[i] = np.concatenate(rotate(X_list, i))[:max_len]
        y_filled[i][:,0] = np.concatenate(rotate(y_list, i))[:max_len]

    print('Filling done.')
    return X_filled, y_filled


def _build_generator(X_list, y_list, batch_size, timesteps, input_size, shuffle, positive_weight):
    """
    Build looping generator of training batches. Will return the generator and the number of batches in each epoch.
    In each epoch, all samples are randomly split into batch_size "chunks", each "chunk" in batch can be trained in parallel.
    Samples in each chunk are shuffled and merged into one whole sequence.
    The whole sequences are separated into batches of given fixed given number of timesteps (protein vectors).
    So the number of batches is defined so that we go over the whole sequence (length of the longest "chunk" sequence divided by the number of timesteps).

    :param X_list: List of samples. Each sample is a matrix/DataFrame of protein domain vectors.
    :param y_list: List of sample outputs.
    :param batch_size: Number of parallel "chunks" in a training batch
    :param timesteps: Number of timesteps (protein domain vectors) in a training batch
    :param input_size: Size of the protein domain vector
    :param shuffle: Whether to shuffle samples within each epoch. If not used, make sure that positive and negative samples are already shuffled in the list.
    :param positive_weight: Weight of positive samples (single number). If provided, a triple of (X_batch, y_batch, weights_batch) are provided
    :return: Tuple of (batch generator, number of batches in each epoch).
    Each batch will contain the X input (batch_size, timesteps, input_size) and y output (batch_size, timesteps, 1)
    """
    if not X_list:
        return _noop, None
    from keras.preprocessing.sequence import pad_sequences
    seq_length = sum([len(X) for X in X_list])
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    num_batches = int(np.ceil(np.ceil(seq_length / batch_size) / timesteps))
    maxlen = num_batches * timesteps
    print('Initializing generator of {} batches from sequence length {}'.format(num_batches, seq_length))

    def generator():
        while True:
            # shuffle the samples
            if shuffle:
                shuffled = np.random.permutation(len(X_list))
            # split samples into batch_size chunks
            X_batches = np.array_split(X_arr[shuffled] if shuffle else X_arr, batch_size)
            y_batches = np.array_split(y_arr[shuffled] if shuffle else y_arr, batch_size)

            # merge the samples in each chunk into one sequence
            X_batches = [np.concatenate(b) if b.size else np.empty(0) for b in X_batches]
            y_batches = [np.concatenate(b) if b.size else np.empty(0) for b in y_batches]

            # pad the sequences with zeros to the length of the longest chunk sequence
            X_batches = pad_sequences(X_batches, maxlen=maxlen, dtype=np.float,
                                                                   padding='post', truncating='post')
            y_batches = pad_sequences(y_batches, maxlen=maxlen, dtype=np.float,
                                                                   padding='post', truncating='post')

            # Reshape array so that it can be indexed as [batch number][chunk][timestep][input feature]
            # This will produce an array of dimension (num_batches, batch_size, timesteps, input_size)
            # And output array of dimension (num_batches, batch_size, timesteps, 1)
            X_batches = np.swapaxes(X_batches.reshape(batch_size, num_batches, timesteps, input_size), 0, 1)
            y_batches = np.swapaxes(y_batches.reshape(batch_size, num_batches, timesteps, 1), 0, 1)

            # print('Generated {}x{} batches: X {}, y {}'.format(num_batches, self.batch_size, X_batches.shape, y_batches.shape))

            if positive_weight:
                # Provide array of weights for each input vector based on the positive weight
                weight_batches = np.ones(y_batches.shape)
                weight_batches[y_batches == 1] = positive_weight
                weight_batches = np.swapaxes(weight_batches.reshape(batch_size, num_batches, timesteps), 0, 1)
                for X_batch, y_batch, weight_batch in zip(X_batches, y_batches, weight_batches):
                    yield X_batch, y_batch, weight_batch
            else:
                for X_batch, y_batch in zip(X_batches, y_batches):
                    yield X_batch, y_batch

    return generator, num_batches

def _count_samples(y_list, klass):
    return np.sum([np.mean(y == klass) for y in y_list])

def _split_matrix_into_batches(X, batch_size):
    if len(X.shape) != 2:
        raise AttributeError('Can only be called on a single 2-dimensional feature matrix.')
    return X.reshape(batch_size, X.shape[0], X.shape[1])

def _pad_matrix_to_be_divisible(X, divisible_by):
    from keras.preprocessing.sequence import pad_sequences
    remainder = X.shape[0] % divisible_by
    if not remainder:
        return X
    maxlen = X.shape[0] + divisible_by - remainder
    return pad_sequences([X], maxlen=maxlen, dtype=np.float, padding='post', truncating='post')[0]


def _get_device(gpus):
    if gpus == 0:
        return tf.device('/cpu:0')
    elif gpus >= 1:
        return tf.device('/device:GPU:0')  # TODO: can we get just the first GPU?
    else:
        raise AttributeError('GPUs has to be an integer >= 0')


def precision(y_true, y_pred):
    """Precision metric.
    
    Only computes a batch-wise average of precision.
    
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    import keras.backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    import keras.backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def auc_roc(y_true, y_pred):
    """
    Defines AUC ROC metric callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
    """
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
