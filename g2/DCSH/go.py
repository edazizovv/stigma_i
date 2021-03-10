#
import time

#
import numpy
import pandas

from matplotlib import pyplot

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn

#
from neuro_kernel import Skeleton


def log_verse(x, x_lag, eps=1e-11):
    return numpy.log((x + eps) / (x_lag + eps))


def no_verse(x, x_lag):
    return x


def log_inverse(x, x_lag, eps=1e-11):
    return numpy.multiply((numpy.exp(x) - eps), (x_lag - eps))


def no_inverse(x, x_lag):
    return x


#
run_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

dt = './data/DailyDelhiClimateTrain.csv'
ds = './data/DailyDelhiClimateTest.csv'

data_train = pandas.read_csv(dt)
data_test = pandas.read_csv(ds)
data_train = data_train.set_index('date')
data_test = data_test.set_index('date')

thresh = data_train.shape[0]

data = pandas.concat((data_train, data_test), axis=0)
data.loc[data['meanpressure'] <= 0, 'meanpressure'] = data['meanpressure'].mean()

target = 'meantemp'
factors = [x for x in data_train.columns.values if target not in x]

scaler = MinMaxScaler()
data[factors] = scaler.fit_transform(data[factors])

data[[x + '_LAG1' for x in data.columns.values]] = data.shift(periods=1)
# data = data.dropna()
data = data.iloc[1:, :].copy()

data_log = data.copy()

for col in [target]:  # [x for x in data.columns.values if 'LAG' not in x]:
    data_log[col] = log_verse(data[col].values, data[col + '_LAG1'].values)
    # data_log[col] = no_verse(data[col].values, data[col + '_LAG1'].values)
data_log[[x for x in data_log.columns.values if 'LAG' in x]] = data_log[
    [x for x in data_log.columns.values if 'LAG' not in x]].shift(periods=1)
# data_log = data_log.dropna()
data_log = data_log.iloc[1:, :].copy()

thresh -= 2

data_train = data_log.iloc[:thresh, :]
data_test = data_log.iloc[thresh:, :]

target = 'meantemp'
# factors = [x for x in data_train.columns.values if 'LAG' in x]
factors = [target]

x_train = data_train[factors].values
y_train = data_train[[target]].values
x_val = data_test[factors].values
y_val = data_test[[target]].values

# activators = [nn.ELU, nn.ELU]
# activators = [nn.LeakyReLU, nn.LeakyReLU]
# activators = [nn.PReLU, nn.PReLU]
# activators = [nn.RReLU, nn.RReLU]
# activators = [nn.SELU, nn.SELU]
# activators = [nn.CELU, nn.CELU]
# activators = [nn.GELU, nn.GELU]
# activators = [nn.SiLU, nn.SiLU]

"""
layers = [nn.Linear] * 2
layers_dimensions = [2, 1]
layers_kwargs = [{}] * 2
activators = [nn.ReLU] * 2
drops = [0.00, 0.00]
verbose = 100
"""
"""
layers = [nn.RNN, nn.Linear]
layers_dimensions = [10, 1]
layers_kwargs = [{'nonlinearity': 'relu'}, {}]
activators = [None, nn.ReLU]
drops = [0.0, 0.0]
verbose = 100
"""
"""
layers = [nn.GRU, nn.Linear]
layers_dimensions = [20, 1]
layers_kwargs = [{}, {}]
activators = [None, nn.ReLU]
drops = [0.05, 0.0]
verbose = 100
"""
"""
layers = [nn.LSTM, nn.Linear]
layers_dimensions = [10, 1]
layers_kwargs = [{}, {}]
activators = [None, nn.ReLU]
drops = [0.1, 0.0]
verbose = 100
"""

verbose = -1
kwg = [
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [10, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.0, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [20, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.0, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [100, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.0, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [10, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [20, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [100, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [10, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [20, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [100, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [10, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.5, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [20, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.5, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear],
     'layers_dimensions': [100, 1],
     'layers_kwargs': [{}, {}],
     'activators': [None, nn.ReLU],
     'drops': [0.5, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [10, 4, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.0, 0.0, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [20, 8, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.0, 0.0, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [100, 40, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.0, 0.0, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [10, 4, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.1, 0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [20, 8, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.1, 0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [100, 40, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.1, 0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [10, 4, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.2, 0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [20, 8, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.2, 0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [100, 40, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.2, 0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [10, 4, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.5, 0.5, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [20, 8, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.5, 0.5, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear],
     'layers_dimensions': [100, 40, 1],
     'layers_kwargs': [{}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU],
     'drops': [0.5, 0.5, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
     'layers_dimensions': [100, 100, 100, 50, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, None, None, None, nn.ReLU],
     'drops': [0.1, 0.1, 0.1, 0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
     'layers_dimensions': [100, 100, 100, 50, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, None, None, None, nn.ReLU],
     'drops': [0.2, 0.2, 0.2, 0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
     'layers_dimensions': [100, 100, 100, 50, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, None, None, None, nn.ReLU],
     'drops': [0.5, 0.5, 0.5, 0.5, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
     'layers_dimensions': [8, 16, 32, 64, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, None, None, None, nn.ReLU],
     'drops': [0.0, 0.0, 0.0, 0.0, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
     'layers_dimensions': [8, 16, 32, 64, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, None, None, None, nn.ReLU],
     'drops': [0.1, 0.1, 0.1, 0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
     'layers_dimensions': [8, 16, 32, 64, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, None, None, None, nn.ReLU],
     'drops': [0.2, 0.2, 0.2, 0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
     'layers_dimensions': [8, 16, 32, 64, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, None, None, None, nn.ReLU],
     'drops': [0.5, 0.5, 0.5, 0.5, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
     'layers_dimensions': [8, 16, 32, 64, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU],
     'drops': [0.0, 0.0, 0.0, 0.0, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
     'layers_dimensions': [8, 16, 32, 64, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU],
     'drops': [0.1, 0.1, 0.1, 0.1, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
     'layers_dimensions': [8, 16, 32, 64, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU],
     'drops': [0.2, 0.2, 0.2, 0.2, 0.0],
     'verbose': verbose,
     'device': device},
    {'layers': [nn.LSTM, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
     'layers_dimensions': [8, 16, 32, 64, 1],
     'layers_kwargs': [{}, {}, {}, {}, {}],
     'activators': [None, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU],
     'drops': [0.5, 0.5, 0.5, 0.5, 0.0],
     'verbose': verbose,
     'device': device},
]

epochs = 1000

# how about wider, wider and wider, huh?

window = 1

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
xx_train_ = numpy.concatenate(xx_train, axis=0)
xx_val_ = numpy.concatenate(xx_val, axis=0)
yy_train_ = numpy.concatenate(yy_train, axis=0)
yy_val_ = numpy.concatenate(yy_val, axis=0)

# yy_train[:20, -1, :]

report = {'No': [], 'r': [], 'k': [], 'kw': [], 'q_train': [], 'q_test': [], 'rs': []}

no = 0
n_repeats = 3
print('n_repeats={0}'.format(n_repeats))
print('n_kwg={0}'.format(len(kwg)))
print('====================================')

for r in range(n_repeats):

    print('r: {0} / {1}'.format(r + 1, n_repeats))

    for k in range(len(kwg)):

        print('\tk: {0} / {1}'.format(k + 1, len(kwg)))

        xx_train = torch.tensor(xx_train_, dtype=torch.float, device=device)
        xx_val = torch.tensor(xx_val_, dtype=torch.float, device=device)
        yy_train = torch.tensor(yy_train_, dtype=torch.float, device=device)
        yy_val = torch.tensor(yy_val_, dtype=torch.float, device=device)

        no += 1

        kwargs = kwg[k]

        # model = Skeleton(layers, layers_dimensions, layers_kwargs, activators, drops, verbose, device=device)
        model = Skeleton(**kwargs)

        optimizer = torch.optim.Adam
        optimizer_kwargs = {'lr': 0.001}
        loss_function = nn.MSELoss()
        model.fit(xx_train, yy_train, xx_val, yy_val, optimizer, optimizer_kwargs, loss_function, epochs=epochs)

        yy_train_hat = model.predict(x=xx_train)
        yy_val_hat = model.predict(x=xx_val)

        if device.type == 'cuda':
            yy_train = yy_train.cpu()
            yy_val = yy_val.cpu()

        # print(yy_train[:, -1, :].numpy().flatten().shape)
        # print(data[target + '_LAG1'].values[window:thresh + 1].shape)

        # y_train_bench = log_inverse(yy_train[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[10:thresh+1])
        # y_train_hat = log_inverse(yy_train_hat[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[10:thresh+1])
        # y_val_bench = log_inverse(yy_val[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[thresh+10:])
        # y_val_hat = log_inverse(yy_val_hat[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[thresh+10:])

        y_train_bench = log_inverse(yy_train[:, -1, :].numpy().flatten(),
                                    data[target + '_LAG1'].values[window:thresh + 1])
        y_train_hat = log_inverse(yy_train_hat[:, -1, :].numpy().flatten(),
                                  data[target + '_LAG1'].values[window:thresh + 1])
        y_val_bench = log_inverse(yy_val[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[thresh + window:])
        y_val_hat = log_inverse(yy_val_hat[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[thresh + window:])
        # y_train_bench = no_inverse(yy_train[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[window-2:thresh+1])
        # y_train_hat = no_inverse(yy_train_hat[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[window-2:thresh+1])
        # y_val_bench = no_inverse(yy_val[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[thresh+window-2:])
        # y_val_hat = no_inverse(yy_val_hat[:, -1, :].numpy().flatten(), data[target + '_LAG1'].values[thresh+window-2:])

        if device.type == 'cuda':
            yy_train = yy_train.cpu()
            yy_val = yy_val.cpu()
        r2_train = r2_score(y_true=y_train_bench, y_pred=y_train_hat)
        r2_val = r2_score(y_true=y_val_bench, y_pred=y_val_hat)
        rmse_train = mean_squared_error(y_true=y_train_bench, y_pred=y_train_hat)
        rmse_val = mean_squared_error(y_true=y_val_bench, y_pred=y_val_hat)

        a = pandas.DataFrame(data={'tt': numpy.arange(start=0, stop=yy_train.shape[0]), 'value': y_train_bench,
                                   'kind': ['true'] * yy_train.shape[0], 'fold': ['train'] * yy_train.shape[0]})
        b = pandas.DataFrame(
            data={'tt': numpy.arange(start=yy_train.shape[0], stop=yy_train.shape[0] + yy_val.shape[0]),
                  'value': y_val_bench, 'kind': ['true'] * yy_val.shape[0], 'fold': ['val'] * yy_val.shape[0]})
        c = pandas.DataFrame(data={'tt': numpy.arange(start=0, stop=yy_train_hat.shape[0]), 'value': y_train_hat,
                                   'kind': ['hat'] * yy_train_hat.shape[0], 'fold': ['train'] * yy_train_hat.shape[0]})
        d = pandas.DataFrame(
            data={'tt': numpy.arange(start=yy_train_hat.shape[0], stop=yy_train_hat.shape[0] + yy_val_hat.shape[0]),
                  'value': y_val_hat, 'kind': ['hat'] * yy_val_hat.shape[0], 'fold': ['val'] * yy_val_hat.shape[0]})
        result = pandas.concat((a, b, c, d), axis=0, ignore_index=True)

        report['No'].append(no)
        report['r'].append(r + 1)
        report['k'].append(k + 1)
        report['kw'].append(kwargs)
        report['q_train'].append(rmse_train)
        report['q_test'].append(rmse_val)
        report['rs'].append(result)

reported = pandas.DataFrame(data={x: report[x] for x in ['No', 'r', 'k', 'q_train', 'q_test']})

run_time = time.time() - run_time

# pyplot.plot(a['tt'].values, a['value'].values, 'black')
# pyplot.plot(b['tt'].values, b['value'].values, 'gray')
# pyplot.plot(c['tt'].values, c['value'].values, 'navy')
# pyplot.plot(d['tt'].values, d['value'].values, 'orange')

# pyplot.plot(a['tt'].values, (a['value'] - c['value']).values, 'black')
# pyplot.plot(b['tt'].values, (b['value'] - d['value']).values, 'navy')

# model.plot()

# RESULTS

# currently the best result for 'meantemp':
# RNNxLinear [10, 1], no drops

# window = 10

# layers = [nn.RNN, nn.Linear]
# layers_dimensions = [10, 1]
# layers_kwargs = [{'nonlinearity': 'relu'}, {}]
# activators = [None, nn.LeakyReLU]
# drops = [0.0, 0.0]
# verbose = 100

# train :: r2   :: 0.9526094936288734
# val   :: r2   :: 0.9377260126640103
# train :: rmse :: 2.4741302
# val   :: rmse :: 2.4960816
