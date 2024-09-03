#
import random

#
import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import torch
from torch import nn


#
from neura import WrappedNN
# from unholy import WrappedNN
from tests import check_up
from recurrent import prepare_inputs, simulate_lags


#
data = pandas.read_csv('./data/dataset.csv')


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#


#
removables = []


target = 'meantemp'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

data = data[[target] + x_factors].dropna()

X = data[[target] + x_factors].values
Y = data[target].values

# X = numpy.random.normal(size=(1000, 1)).cumsum(axis=0)
# Y = X.flatten()

thresh = 0.5
start = 0
mid = int(X.shape[0] * thresh)
end = -1
X_train, X_test, Y_train, Y_test = X[start:mid, :], X[mid:end, :], Y[start:mid], Y[mid:end]

_X_train = X_train.copy()
_X_test = X_test.copy()
_Y_train = Y_train.copy()
_Y_test = Y_test.copy()


X_train = pandas.DataFrame(X_train).pct_change().values[1:]
Y_train = pandas.DataFrame(Y_train).pct_change().values[1:]
X_test = pandas.DataFrame(X_test).pct_change().values[1:]
Y_test = pandas.DataFrame(Y_test).pct_change().values[1:]

for j in range(X.shape[1]):
    X_train[~numpy.isfinite(X_train[:, j]), j] = numpy.ma.masked_invalid(X_train[:, j]).max()
    X_test[~numpy.isfinite(X_test[:, j]), j] = numpy.ma.masked_invalid(X_test[:, j]).max()

'''
X_train = pandas.DataFrame(X_train).diff().values[1:]
Y_train = pandas.DataFrame(Y_train).diff().values[1:]
X_test = pandas.DataFrame(X_test).diff().values[1:]
Y_test = pandas.DataFrame(Y_test).diff().values[1:]
'''
'''
X_train = pandas.DataFrame(X_train).values[1:]
Y_train = pandas.DataFrame(Y_train).values[1:]
X_test = pandas.DataFrame(X_test).values[1:]
Y_test = pandas.DataFrame(Y_test).values[1:]
'''
#
"""
thresh = 0.01
values = mutual_info_classif(X=X_train, y=Y_train, discrete_features='auto')
fs_mask = values >= thresh
"""
"""
thresh = 0.05
values = numpy.array([spearmanr(a=X_train[:, j], b=Y_train)[0] for j in range(X_train.shape[1])])
fs_mask = numpy.abs(values) >= thresh
"""
"""
thresh = 0.05
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[0] for j in range(X_train.shape[1])])
fs_mask = numpy.abs(values) >= thresh
"""
"""
alpha = 0.05
values = numpy.array([spearmanr(a=X_train[:, j], b=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha
"""
"""
alpha = 0.05
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha
"""
"""
X_train = X_train[:, fs_mask]
X_test = X_test[:, fs_mask]
"""


"""
scaler = StandardScaler()
scaler.fit(X=X_train)
X_train_ = scaler.transform(X_train)
X_test_ = scaler.transform(X_test)
"""
# """
X_train_ = X_train
X_test_ = X_test
# """

"""
# proj_rate = 0.50  # 0.75   0.5   0.25  'mle'
# njv = int(X_train_.shape[1] * proj_rate)
njv = 'mle'
# njv = 0.75  # 0.75  0.5  0.25
projector = PCA(n_components=njv, svd_solver='full', random_state=rs)
projector.fit(X=X_train_)
X_train_ = projector.transform(X_train_)
X_test_ = projector.transform(X_test_)
"""
"""
proj_rate = 0.50  # 0.75   0.5   0.25
gamma = 0.1  # None  0.001  0.1  1
njv = int(X_train_.shape[1] * proj_rate)
projector = KernelPCA(n_components=njv, random_state=rs, remove_zero_eig=True, gamma=gamma, kernel='rbf')
projector.fit(X=X_train_)
X_train_ = projector.transform(X_train_)
X_test_ = projector.transform(X_test_)
"""

from reviser_model import TSNumericKerasReviser as ReviserModel
import tensorflow as tf
from tensorflow import keras

Y_train_ = Y_train.reshape(-1, 1)
Y_test_ = Y_test.reshape(-1, 1)

window = 40
# """
model_kwargs = {
    'dims': [256, 128, 64, 1],
    'layers': [keras.layers.LSTM, keras.layers.Dense, keras.layers.Dense, keras.layers.Dense],
    'window': window,
    'activators': ['relu', 'relu', 'relu', None],  # [keras.layers.LeakyReLU(), keras.layers.LeakyReLU()],
    'drops': [0.2, 0.2, 0.2, 0.0],
    'batchnorms': [True, True, True, False],
    'n_epochs': 100,
    'optimiser': tf.keras.optimizers.Adam,                # AdamW
    'optimiser_kwargs': {'learning_rate': 0.01},          # weight_decay=0.001
    'loss_function': 'mse',
}
# """
"""
model_kwargs = {
    'dims': [20, 2, 1],
    'layers': [keras.layers.LSTM, keras.layers.Dense, keras.layers.Dense],
    'window': window,
    'activators': ['relu', 'relu', None],  # [keras.layers.LeakyReLU(), keras.layers.LeakyReLU()],
    'drops': [0.0, 0.0, 0.0],
    'batchnorms': [True, True, False],
    'n_epochs': 100,
    'optimiser': tf.keras.optimizers.Adam,                # AdamW
    'optimiser_kwargs': {'learning_rate': 0.01},          # weight_decay=0.001
    'loss_function': 'mse',
}
"""
model = ReviserModel(**model_kwargs)

model.fit(X_train=X_train_, Y_train=Y_train_, X_val=X_train_, Y_val=Y_train_)
y_hat_train = model.predict(X=X_train_, Y=Y_train_)
y_hat_test = model.predict(X=X_test_, Y=Y_test_)

# y_hat_train = model.predict(X=xx_train, Y_base=_Y_train[window:-1].reshape(-1, 1))
# y_hat_test = model.predict(X=xx_test, Y_base=_Y_test[window:-1].reshape(-1, 1))

yy_train = Y_train_[window-1:] + 1
# yy_train[0] = _Y_train[window+1]
yy_train = yy_train.flatten() * _Y_train[window-1:-1]  # yy_train.cumprod(axis=0)

y_hat_train = y_hat_train + 1
# y_hat_train[0] = _Y_train[window+1]
y_hat_train = y_hat_train.flatten() * _Y_train[window-1:-1]  # y_hat_train.cumprod(axis=0)

yy_test = Y_test_[window-1:] + 1
# yy_test[0] = _Y_test[window+1]
yy_test = yy_test.flatten() * _Y_test[window-1:-1] # yy_test.cumprod(axis=0)

y_hat_test = y_hat_test + 1
# y_hat_test[0] = _Y_test[window+1]
y_hat_test = y_hat_test.flatten() * _Y_test[window-1:-1] # y_hat_test.cumprod(axis=0)

'''
yy_train = yy_train.numpy().flatten() + _Y_train[window:-1]  # yy_train.cumprod(axis=0)

y_hat_train = y_hat_train.flatten() + _Y_train[window:-1]  # y_hat_train.cumprod(axis=0)

yy_test = yy_test.numpy().flatten() + _Y_test[window:-1] # yy_test.cumprod(axis=0)

y_hat_test = y_hat_test.flatten() + _Y_test[window:-1] # y_hat_test.cumprod(axis=0)
'''
'''
yy_train = yy_train.numpy().flatten()

y_hat_train = y_hat_train.flatten()

yy_test = yy_test.numpy().flatten()

y_hat_test = y_hat_test.flatten()
'''

results_train = check_up(yy_train, y_hat_train, None, X_train_)
results_test = check_up(yy_test, y_hat_test, None, X_test_)

'''
mody = Mody(model)
results_train = check_up(yy_train[1:], yy_train[:-1], mody, X_train_)
results_test = check_up(yy_test[1:], yy_test[:-1], mody, X_test_)
'''
results_train['sample'] = 'train'
results_test['sample'] = 'test'

results_train = pandas.DataFrame(pandas.Series(results_train))
results_test = pandas.DataFrame(pandas.Series(results_test))

"""

# joblib.dump(model, filename='./model_ex12.pkl')
results_train.T.to_csv('./reported.csv', mode='a', header=False)
results_test.T.to_csv('./reported.csv', mode='a', header=False)

fig, ax = pyplot.subplots(2, 2, sharex='col', sharey='col')
ax[0, 0].plot(range(yy_train.shape[0]), yy_train.flatten() - y_hat_train.flatten(), color='lightgrey')
ax[1, 0].plot(range(yy_test.shape[0]), yy_test.flatten() - y_hat_test.flatten(), color='orange')
ax[0, 1].hist(yy_train.flatten() - y_hat_train.flatten(), color='lightgrey', bins=100, density=True)
ax[1, 1].hist(yy_test.flatten() - y_hat_test.flatten(), color='orange', bins=100, density=True)

pyplot.plot(range(yy_train.shape[0]), yy_train, 'lightgrey', range(yy_train.shape[0]), y_hat_train, 'orange')
pyplot.plot(range(yy_test.shape[0]), yy_test, 'lightgrey', range(yy_test.shape[0]), y_hat_test, 'orange')

pyplot.plot(range(yy_train[:100].shape[0]), yy_train[:100], 'lightgrey', range(yy_train[:100].shape[0]), y_hat_train[:100], 'orange')
pyplot.plot(range(yy_test[:100].shape[0]), yy_test[:100], 'lightgrey', range(yy_test[:100].shape[0]), y_hat_test[:100], 'orange')

"""
'''
pyplot.plot(range(yy_train.shape[0]-1), pandas.Series(yy_train).pct_change()[1:].values, 'lightgrey', range(yy_train.shape[0]-1), pandas.Series(y_hat_train).pct_change()[1:].values, 'orange')
pyplot.plot(range(yy_test.shape[0]-1), pandas.Series(yy_test).pct_change()[1:].values, 'lightgrey', range(yy_test.shape[0]-1), pandas.Series(y_hat_test).pct_change()[1:].values, 'orange')

pyplot.plot(range(yy_train[:100].shape[0]), pandas.Series(yy_train).pct_change()[1:].values[:100], 'lightgrey', range(yy_train[:100].shape[0]), pandas.Series(y_hat_train).pct_change()[1:].values[:100], 'orange')
pyplot.plot(range(yy_test[:100].shape[0]), pandas.Series(yy_test).pct_change()[1:].values[:100], 'lightgrey', range(yy_test[:100].shape[0]), pandas.Series(y_hat_test).pct_change()[1:].values[:100], 'orange')

'''