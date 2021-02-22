#


#
import numpy
import pandas

from matplotlib import pyplot

from sklearn.metrics import r2_score, mean_squared_error

import torch
from torch import nn


#
from neuro_kernel import Skeleton


def log_verse(x, x_lag, eps=1e-11):
    return numpy.log((x + eps) / (x_lag + eps))


def log_inverse(x, x_lag, eps=1e-11):
    return numpy.multiply((numpy.exp(x) - eps), (x_lag - eps))


#
dt = './data/DailyDelhiClimateTrain.csv'
ds = './data/DailyDelhiClimateTest.csv'

data_train = pandas.DataFrame(data={'meanpressure': numpy.arange(start=100, stop=200),
                                    'meantemp': numpy.arange(start=100, stop=200)})
data_test = pandas.DataFrame(data={'meanpressure': numpy.arange(start=200, stop=250),
                                    'meantemp': numpy.arange(start=200, stop=250)})

thresh = data_train.shape[0]

data = pandas.concat((data_train, data_test), axis=0, ignore_index=True)
data.loc[data['meanpressure'] <= 0, 'meanpressure'] = data['meanpressure'].mean()

data[[x + '_LAG1' for x in data.columns.values]] = data.shift(periods=1)
# data = data.dropna()
data = data.iloc[1:, :].copy()

data_log = data.copy()
for col in [x for x in data.columns.values if 'LAG' not in x]:
    data_log[col] = log_verse(data[col].values, data[col + '_LAG1'].values)
data_log[[x for x in data_log.columns.values if 'LAG' in x]] = data_log[[x for x in data_log.columns.values if 'LAG' not in x]].shift(periods=1)
# data_log = data_log.dropna()
data_log = data_log.iloc[1:, :].copy()

thresh -= 2

data_train = data_log.iloc[:thresh, :]
data_test = data_log.iloc[thresh:, :]

target = 'meantemp'
factors = [x for x in data_train.columns.values if 'LAG' in x]


x_train = data_train[factors].values
y_train = data_train[[target]].values
x_val = data_test[factors].values
y_val = data_test[[target]].values

window = 10

xx_train = []
xx_val = []
yy_train = []
yy_val = []
for j in range(x_train.shape[0] - window + 1):
    xx_train.append(x_train[j:j + window, :].reshape(1, window, x_train.shape[1]))
    yy_train.append(y_train[j:j + window].reshape(1, window, 1))
for j in range(x_val.shape[0] - window + 1):
    xx_val.append(x_val[j:j + window, :].reshape(1, window, x_val.shape[1]))
    yy_val.append(y_val[j:j + window].reshape(1, window, 1))
xx_train = numpy.concatenate(xx_train, axis=0)
xx_val = numpy.concatenate(xx_val, axis=0)
yy_train = numpy.concatenate(yy_train, axis=0)
yy_val = numpy.concatenate(yy_val, axis=0)
xx_train = torch.tensor(xx_train, dtype=torch.float)
xx_val = torch.tensor(xx_val, dtype=torch.float)
yy_train = torch.tensor(yy_train, dtype=torch.float)
yy_val = torch.tensor(yy_val, dtype=torch.float)
