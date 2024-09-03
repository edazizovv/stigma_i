import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from sklearn.metrics import r2_score

from zipfile import ZipFile
import os
from reviser_model import TSNumericKerasReviser as ReviserModel


uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "./jena_climate_2009_2016.csv"

dataframe = pd.read_csv(csv_path)


excluded = ['Date Time']
target = 'Tpot (K)'
x_factors = [x for x in dataframe.columns.values if x not in excluded]

thresh = 0.5
start_point = 0
mid_point = int(dataframe.shape[0] * thresh)
end_point = -1

window = 10

X, y = dataframe[x_factors].iloc[:-window].values, dataframe[[target]].iloc[window:].values

X_train, X_test, y_train, y_test = X[start_point:mid_point], X[mid_point:end_point], y[start_point:mid_point], y[mid_point:end_point]

model_kwargs = {
    'dims': [32, 1],
    'layers': [keras.layers.LSTM, keras.layers.Dense],
    'window': 10,
    'activators': ['tanh', None],  # [keras.layers.LeakyReLU(), keras.layers.LeakyReLU()],
    'drops': [0.0, 0.0],
    'batchnorms': [False, False],
    'n_epochs': 100,
    'optimiser': tf.keras.optimizers.Adam,
    'optimiser_kwargs': {'learning_rate': 0.001},
    'loss_function': 'mse',
}
model = ReviserModel(**model_kwargs)

model.fit(X_train=X_train, Y_train=y_train, X_val=X_test, Y_val=y_test)
y_hat_train = model.predict(X=X_train, Y=y_train)
y_hat_test = model.predict(X=X_test, Y=y_test)

mx_train = r2_score(y_true=y_train[window-1:], y_pred=y_hat_train)
mx_test = r2_score(y_true=y_test[window-1:], y_pred=y_hat_test)

fig, ax = pyplot.subplots(2, 1)
ax[0].plot(range(y_train.shape[0]), y_train, 'navy', range(y_hat_train.shape[0]), y_hat_train, 'orange')
ax[1].plot(range(y_test.shape[0]), y_test, 'navy', range(y_hat_test.shape[0]), y_hat_test, 'orange')

# keras.utils.plot_model(self.model, show_shapes=True, rankdir="LR")
