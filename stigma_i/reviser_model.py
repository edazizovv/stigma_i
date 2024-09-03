#


#
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers


#


#
# https://keras.io/getting_started/intro_to_keras_for_engineers/
# https://keras.io/getting_started/intro_to_keras_for_researchers/
# https://keras.io/examples/structured_data/structured_data_classification_from_scratch/#feature-preprocessing-with-keras-layers

class LinearCSNumericKerasReviser:

    def __init__(self, dims, activators, drops, batchnorms, n_epochs, optimiser, optimiser_kwargs, loss_function):
        self.dims = dims
        self.activators = activators
        self.drops = drops
        self.batchnorms = batchnorms
        self.n_epochs = n_epochs
        self._optimiser = optimiser
        self._optimiser_kwargs = optimiser_kwargs
        self.optimiser = optimiser(**optimiser_kwargs)
        self.loss_function = loss_function

    def initiate_features(self, x):
        # all_inputs = keras.Input(shape=(x.shape[1],), name='vex')
        all_inputs = [
            keras.Input(shape=(1,)) for _ in range(x.shape[1])
        ]

        return all_inputs

    def initiate_layers(self, x, y):
        # possibly switch to sequential:
        # https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras

        all_inputs = self.initiate_features(x=x)
        all_features = keras.layers.concatenate(all_inputs)

        z = all_features  # all_inputs
        for j in range(len(self.dims)):
            z = keras.layers.Dense(self.dims[j], activation=self.activators[j])(z)
            # z = self.activators[j](z)
            z = keras.layers.Dropout(self.drops[j])(z)
            if self.batchnorms[j]:
                z = keras.layers.BatchNormalization()(z)
        self.model = keras.Model(all_inputs, z)
        self.model.compile(self.optimiser, self.loss_function)

    def fit(self, X_train, Y_train, X_val, Y_val):

        self.initiate_layers(x=X_train, y=Y_train)

        xx_train = {'input_{0}'.format(j+1): X_train[:, [j]] for j in range(X_train.shape[1])}
        xx_val = {'input_{0}'.format(j+1): X_val[:, [j]] for j in range(X_val.shape[1])}

        dataset_train = tf.data.Dataset.from_tensor_slices((xx_train, Y_train))
        dataset_val = tf.data.Dataset.from_tensor_slices((xx_val, Y_val))

        self.model.fit(dataset_train, epochs=self.n_epochs, validation_data=dataset_val)

    def predict(self, X, Y):

        xx = {'input_{0}'.format(j+1): X[:, [j]] for j in range(X.shape[1])}

        dataset = tf.data.Dataset.from_tensor_slices((xx, Y))
        predicted = self.model.predict(dataset)

        return predicted


class TSNumericKerasReviser:

    def __init__(self, dims, layers, activators, drops, batchnorms, n_epochs, optimiser, optimiser_kwargs, loss_function, window):
        self.dims = dims
        self._layers = layers
        self.activators = activators
        self.drops = drops
        self.batchnorms = batchnorms
        self.n_epochs = n_epochs
        self._optimiser = optimiser
        self._optimiser_kwargs = optimiser_kwargs
        self.optimiser = optimiser(**optimiser_kwargs)
        self.loss_function = loss_function
        self.window = window

    def initiate_features(self, ds):
        for batch in ds.take(1):
            inputs, targets = batch
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))  # 1 2 in the batched case

        return inputs

    def initiate_layers(self, x, y):
        # possibly switch to sequential:
        # https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras

        ds = keras.preprocessing.timeseries_dataset_from_array(
            x,
            y,
            sequence_length=self.window,
            sampling_rate=1,
            batch_size=128,
        )

        inputs = self.initiate_features(ds=ds)

        z = inputs
        for j in range(len(self.dims)):
            z = self._layers[j](self.dims[j], activation=self.activators[j])(z)
            # z = self.activators[j](z)
            z = keras.layers.Dropout(self.drops[j])(z)
            if self.batchnorms[j]:
                z = keras.layers.BatchNormalization()(z)
        self.model = keras.Model(inputs, z)
        self.model.compile(self.optimiser, self.loss_function)

    def fit(self, X_train, Y_train, X_val, Y_val):

        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            X_train,
            Y_train,
            sequence_length=self.window,
            sampling_rate=1,
            batch_size=128,
        )

        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            X_val,
            Y_val,
            sequence_length=self.window,
            sampling_rate=1,
            batch_size=128,
        )

        self.initiate_layers(x=X_train, y=Y_train)

        self.model.fit(dataset_train, epochs=self.n_epochs, validation_data=dataset_val)

    def predict(self, X, Y):

        dataset = keras.preprocessing.timeseries_dataset_from_array(
            X,
            Y,
            sequence_length=self.window,
            sampling_rate=1,
            batch_size=128,
        )

        predicted = self.model.predict(dataset)

        return predicted
