#


#
import pandas
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


#
from reviser_model import LinearCSNumericKerasReviser as ReviserModel


#
rs = 997

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pandas.read_csv(file_url)

excluded = ['thal']
target = 'target'
x_factors = [x for x in dataframe.columns.values if x not in excluded + [target]]

X, y = dataframe[x_factors].values, dataframe[[target]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=rs)

model_kwargs = {
    'dims': [32, 1],
    'activators': [keras.layers.LeakyReLU(), 'sigmoid'],
    'drops': [0.5, 0.0],
    'batchnorms': [False, False],
    'n_epochs': 50,
    'optimiser': tf.keras.optimizers.Adam,
    'optimiser_kwargs': {'learning_rate': 0.001},
    'loss_function': 'binary_crossentropy',
}
model = ReviserModel(**model_kwargs)

model.fit(X_train=X_train, Y_train=y_train, X_val=X_test, Y_val=y_test)
y_hat_train = model.predict(X=X_train, Y=y_train)
y_hat_test = model.predict(X=X_test, Y=y_test)

y_hat_train = (y_hat_train >= 0.5).astype(dtype=int)
y_hat_test = (y_hat_test >= 0.5).astype(dtype=int)

mx_train = confusion_matrix(y_true=y_train, y_pred=y_hat_train)
mx_test = confusion_matrix(y_true=y_test, y_pred=y_hat_test)

# keras.utils.plot_model(self.model, show_shapes=True, rankdir="LR")
