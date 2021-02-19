#


#
import numpy
import pandas

import torch
from torch import nn


#
from neuro_kernel import Skeleton


#
dt = './data/DailyDelhiClimateTrain.csv'
ds = './data/DailyDelhiClimateTest.csv'

data_train = pandas.read_csv(dt)
data_test = pandas.read_csv(ds)

data_train = data_train.set_index('date')
data_test = data_test.set_index('date')

target = 'meantemp'
factors = [x for x in data_train.columns.values if x != target]


x_train = data_train[factors].values
y_train = data_train[[target]].values
x_val = data_train[factors].values
y_val = data_train[[target]].values


layers = [nn.Linear, nn.Linear]
layers_dimensions = [2, 1]
layers_kwargs = [{}, {}]
activators = [nn.ReLU, nn.ReLU]
drops = [0.0, 0.0]
verbose = 25

"""
layers = [nn.RNN, nn.Linear]
layers_dimensions = [2, 1]
layers_kwargs = [{'nonlinearity': 'relu'}, {}]
activators = [None, nn.ReLU]
drops = [0.0, 0.0]
verbose = 25
"""
"""
layers = [nn.GRU, nn.Linear]
layers_dimensions = [2, 1]
layers_kwargs = [{}, {}]
activators = [None, nn.ReLU]
drops = [0.0, 0.0]
verbose = 25
"""
"""
layers = [nn.LSTM, nn.Linear]
layers_dimensions = [2, 1]
layers_kwargs = [{}, {}]
activators = [None, nn.ReLU]
drops = [0.0, 0.0]
verbose = 25
"""

model = Skeleton(layers, layers_dimensions, layers_kwargs, activators, drops, verbose)

optimizer = torch.optim.Adam
optimizer_kwargs = {'lr': 0.001}
loss_function = nn.MSELoss()

torch.tensor(x_train, dtype=torch.float)

window = 10

xx_train = []
xx_val = []
yy_train = []
yy_val = []
for j in range(x_train.shape[0] - window):
    xx_train.append(x_train[-j - window - 1:-j - 1, :].reshape(1, window, x_train.shape[1]))
    yy_train.append(y_train[-j - window - 1:-j - 1].reshape(1, window, 1))
for j in range(x_val.shape[0] - window):
    xx_val.append(x_val[-j - window - 1:-j - 1, :].reshape(1, window, x_val.shape[1]))
    yy_val.append(y_val[-j - window - 1:-j - 1].reshape(1, window, 1))
xx_train = numpy.concatenate(xx_train, axis=0)
xx_val = numpy.concatenate(xx_val, axis=0)
yy_train = numpy.concatenate(yy_train, axis=0)
yy_val = numpy.concatenate(yy_val, axis=0)
xx_train = torch.tensor(xx_train, dtype=torch.float)
xx_val = torch.tensor(xx_val, dtype=torch.float)
yy_train = torch.tensor(yy_train, dtype=torch.float)
yy_val = torch.tensor(yy_val, dtype=torch.float)

model.fit(xx_train, yy_train, xx_val, yy_val, optimizer, optimizer_kwargs, loss_function, epochs=500)
