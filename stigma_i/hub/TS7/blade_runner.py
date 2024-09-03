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
pyplot.style.use('dark_background')


data = pandas.read_csv('./data/dataset.csv')
data = data.set_index('date')


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#


#
removables = []

target = 'USAGE'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

data = data[[target] + x_factors]

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

'''
X_train = pandas.DataFrame(X_train).pct_change().fillna(0).values[1:]
Y_train = pandas.DataFrame(Y_train).pct_change().fillna(0).values[1:]
X_test = pandas.DataFrame(X_test).pct_change().fillna(0).values[1:]
Y_test = pandas.DataFrame(Y_test).pct_change().fillna(0).values[1:]
'''

X_train = pandas.DataFrame(X_train).diff().fillna(0).values[1:]
Y_train = pandas.DataFrame(Y_train).diff().fillna(0).values[1:]
X_test = pandas.DataFrame(X_test).diff().fillna(0).values[1:]
Y_test = pandas.DataFrame(Y_test).diff().fillna(0).values[1:]

for j in range(X.shape[1]):
    X_train[numpy.isinf(X_train[:, j]), j] = numpy.ma.masked_invalid(X_train[:, j]).max()
    X_test[numpy.isinf(X_test[:, j]), j] = numpy.ma.masked_invalid(X_test[:, j]).max()
Y_train[numpy.isinf(Y_train)] = numpy.ma.masked_invalid(Y_train).max()
Y_test[numpy.isinf(Y_test)] = numpy.ma.masked_invalid(Y_test).max()


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


Y_train_ = Y_train.reshape(-1, 1)
Y_test_ = Y_test.reshape(-1, 1)

X_train_ = torch.tensor(X_train_, dtype=torch.float)
Y_train_ = torch.tensor(Y_train_, dtype=torch.float)
X_test_ = torch.tensor(X_test_, dtype=torch.float)
Y_test_ = torch.tensor(Y_test_, dtype=torch.float)

from neura import Test

window = 5

# xx_train, yy_train = prepare_inputs(X_train_, Y_train, window)
# xx_test, yy_test = prepare_inputs(X_test_, Y_test, window)
xx_train, yy_train = simulate_lags(X_train_, Y_train, window)
xx_test, yy_test = simulate_lags(X_test_, Y_test, window)

# """
nn_kwargs = {'layers': [nn.Linear, nn.Linear, nn.Linear],
             'layers_dimensions': [20, 2, 1],
             # 'layers_kwargs': [{'batch_first': True}, {}, {}],
             'layers_kwargs': [{}, {}, {}],
             'batchnorms': [None, nn.BatchNorm1d, None],  # nn.BatchNorm1d
             'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
             'interdrops': [0.0, 0.0, 0.0],
             'optimiser': torch.optim.Adamax,  # Adamax / AdamW / SGD
             'optimiser_kwargs': {'lr': 0.05,
                                  # 'weight_decay': 0.001,
                                  # 'momentum': 0.9,
                                  # 'nesterov': True,
                                  },

             'loss_function': nn.MSELoss,
             'epochs': 2000,
             #  'device': device,
             }

model = WrappedNN(**nn_kwargs)
# """
"""
nn_kwargs = {'n_hidden': 10,
             'epochs': 1000,
             'optimiser': torch.optim.Adamax,
             'optimiser_kwargs': {'lr': 0.05},
             'loss_function': nn.MSELoss
             }

model = Test(**nn_kwargs)
"""

model.fit(X_train=xx_train, Y_train=yy_train.reshape(-1, 1), X_val=xx_test, Y_val=yy_test.reshape(-1, 1))
# model.fit(X_train=xx_train, Y_train=yy_train.reshape(-1, 1), Y_train_base=torch.tensor(_Y_train[window:-1].reshape(-1, 1), dtype=torch.float),
#           X_val=xx_test, Y_val=yy_test.reshape(-1, 1), Y_val_base=torch.tensor(_Y_test[window:-1].reshape(-1, 1), dtype=torch.float))

y_hat_train = model.predict(X=xx_train)
y_hat_test = model.predict(X=xx_test)

# y_hat_train = model.predict(X=xx_train, Y_base=_Y_train[window:-1].reshape(-1, 1))
# y_hat_test = model.predict(X=xx_test, Y_base=_Y_test[window:-1].reshape(-1, 1))


class Mody:
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        # X, y = prepare_inputs(X, y, window)
        X, y = simulate_lags(X, y, window)
        self.model.fit(X_train=X, Y_train=y, # Y_train_base=torch.ones(size=y.shape),
                       X_val=X, Y_val=y) # , Y_val_base=torch.ones(size=y.shape))
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float)
        # X, _ = prepare_inputs(X, numpy.ones(shape=(X.shape[0], 1)), window)
        X, _ = simulate_lags(X, numpy.ones(shape=(X.shape[0], 1)), window)
        return self.model.predict(X=X) # , Y_base=numpy.ones(shape=(X.shape[0], 1)))

'''
yy_train = yy_train + 1
# yy_train[0] = _Y_train[window+1]
yy_train = yy_train.numpy().flatten() * _Y_train[window:-1]  # yy_train.cumprod(axis=0)

y_hat_train = y_hat_train + 1
# y_hat_train[0] = _Y_train[window+1]
y_hat_train = y_hat_train.flatten() * _Y_train[window:-1]  # y_hat_train.cumprod(axis=0)

yy_test = yy_test + 1
# yy_test[0] = _Y_test[window+1]
yy_test = yy_test.numpy().flatten() * _Y_test[window:-1] # yy_test.cumprod(axis=0)

y_hat_test = y_hat_test + 1
# y_hat_test[0] = _Y_test[window+1]
y_hat_test = y_hat_test.flatten() * _Y_test[window:-1] # y_hat_test.cumprod(axis=0)
'''
# '''
yy_train = yy_train.numpy().flatten() + _Y_train[window:-1]  # yy_train.cumprod(axis=0)

y_hat_train = y_hat_train.flatten() + _Y_train[window:-1]  # y_hat_train.cumprod(axis=0)

yy_test = yy_test.numpy().flatten() + _Y_test[window:-1] # yy_test.cumprod(axis=0)

y_hat_test = y_hat_test.flatten() + _Y_test[window:-1] # y_hat_test.cumprod(axis=0)
# '''
'''
yy_train = yy_train.numpy().flatten()

y_hat_train = y_hat_train.flatten()

yy_test = yy_test.numpy().flatten()

y_hat_test = y_hat_test.flatten()
'''

mody = Mody(model)
results_train = check_up(yy_train, y_hat_train, mody, X_train_)
results_test = check_up(yy_test, y_hat_test, mody, X_test_)

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