#


#
import numpy
import pandas

from matplotlib import pyplot

from sklearn.metrics import r2_score, mean_squared_error

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM


#
from neuro_kernel import Skeleton


def log_verse(x, x_lag, eps=1e-11):
    return numpy.log((x + eps) / (x_lag + eps))


def log_inverse(x, x_lag, eps=1e-11):
    return numpy.multiply((numpy.exp(x) - eps), (x_lag - eps))


#
dt = './data/DailyDelhiClimateTrain.csv'
ds = './data/DailyDelhiClimateTest.csv'

data_train = pandas.read_csv(dt)
data_test = pandas.read_csv(ds)
data_train = data_train.set_index('date')
data_test = data_test.set_index('date')

thresh = data_train.shape[0]

data = pandas.concat((data_train, data_test), axis=0)
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

window = 6

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
yy_train = numpy.concatenate(yy_train, axis=0)[:, -1, :].flatten()
yy_val = numpy.concatenate(yy_val, axis=0)[:, -1, :].flatten()
"""
model = Sequential()
model.add(LSTM(10, input_shape=(window, xx_train.shape[2])))
model.add(Dense(1))
"""

model = Sequential([
    LSTM(8,input_shape=(window, xx_train.shape[2])),
    Dense(16,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

epochs = 100
# batch_size = 64
batch_size = 16

history = model.fit(xx_train, yy_train, epochs=epochs,
                    batch_size=batch_size)

yy_train_hat = model.predict(x=xx_train).flatten()
yy_val_hat = model.predict(x=xx_val).flatten()

y_train_bench = log_inverse(yy_train, data[target + '_LAG1'].values[window:thresh+1])
y_train_hat = log_inverse(yy_train_hat, data[target + '_LAG1'].values[window:thresh+1])
y_val_bench = log_inverse(yy_val, data[target + '_LAG1'].values[thresh+window:])
y_val_hat = log_inverse(yy_val_hat, data[target + '_LAG1'].values[thresh+window:])


r2_train = r2_score(y_true=y_train_bench, y_pred=y_train_hat)
r2_val = r2_score(y_true=y_val_bench, y_pred=y_val_hat)
rmse_train = mean_squared_error(y_true=y_train_bench, y_pred=y_train_hat)
rmse_val = mean_squared_error(y_true=y_val_bench, y_pred=y_val_hat)

a = pandas.DataFrame(data={'tt': numpy.arange(start=0, stop=yy_train.shape[0]), 'value': y_train_bench, 'kind': ['true'] * yy_train.shape[0], 'fold': ['train'] * yy_train.shape[0]})
b = pandas.DataFrame(data={'tt': numpy.arange(start=yy_train.shape[0], stop=yy_train.shape[0]+yy_val.shape[0]), 'value': y_val_bench, 'kind': ['true'] * yy_val.shape[0], 'fold': ['val'] * yy_val.shape[0]})
c = pandas.DataFrame(data={'tt': numpy.arange(start=0, stop=yy_train_hat.shape[0]), 'value': y_train_hat, 'kind': ['hat'] * yy_train_hat.shape[0], 'fold': ['train'] * yy_train_hat.shape[0]})
d = pandas.DataFrame(data={'tt': numpy.arange(start=yy_train_hat.shape[0], stop=yy_train_hat.shape[0]+yy_val_hat.shape[0]), 'value': y_val_hat, 'kind': ['hat'] * yy_val_hat.shape[0], 'fold': ['val'] * yy_val_hat.shape[0]})
result = pandas.concat((a, b, c, d), axis=0, ignore_index=True)

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

