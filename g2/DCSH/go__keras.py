#


#
import numpy
import pandas

from matplotlib import pyplot

from sklearn.metrics import r2_score, mean_squared_error

from keras.models import Sequential
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

data[[x + '_LAG1' for x in data.columns.values]] = data.shift(periods=1)
data = data.dropna()

data_log = data.copy()
for col in [x for x in data.columns.values if 'LAG' not in x]:
    data_log[col] = log_verse(data[col].values, data[col + '_LAG1'].values)
data_log[[x for x in data_log.columns.values if 'LAG' in x]] = data_log[[x for x in data_log.columns.values if 'LAG' not in x]].shift(periods=1)
data_log = data_log.dropna()

data_train = data.iloc[:thresh, :]
data_test = data.iloc[thresh:, :]

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
for j in range(x_train.shape[0] - window):
    xx_train.append(x_train[-j - window - 1:-j - 1, :].reshape(1, window, x_train.shape[1]))
    yy_train.append(y_train[-j - window - 1:-j - 1].reshape(1, window, 1))
for j in range(x_val.shape[0] - window):
    xx_val.append(x_val[-j - window - 1:-j - 1, :].reshape(1, window, x_val.shape[1]))
    yy_val.append(y_val[-j - window - 1:-j - 1].reshape(1, window, 1))
xx_train = numpy.concatenate(xx_train, axis=0)
xx_val = numpy.concatenate(xx_val, axis=0)
yy_train = numpy.concatenate(yy_train, axis=0)[:, -1, :].flatten()
yy_val = numpy.concatenate(yy_val, axis=0)[:, -1, :].flatten()

model = Sequential()
model.add(LSTM(10, input_shape=(10, xx_train.shape[2])))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

epochs = 200
batch_size = 64

history = model.fit(xx_train, yy_train, epochs=epochs,
                    batch_size=batch_size)


yy_train_hat = model.predict(x=xx_train).flatten()
yy_val_hat = model.predict(x=xx_val).flatten()

r2_train = r2_score(y_true=yy_train, y_pred=yy_train_hat)
r2_val = r2_score(y_true=yy_val, y_pred=yy_val_hat)
rmse_train = mean_squared_error(y_true=yy_train, y_pred=yy_train_hat)
rmse_val = mean_squared_error(y_true=yy_val, y_pred=yy_val_hat)

a = pandas.DataFrame(data={'tt': numpy.arange(start=0, stop=yy_train.shape[0]), 'value': yy_train, 'kind': ['true'] * yy_train.shape[0], 'fold': ['train'] * yy_train.shape[0]})
b = pandas.DataFrame(data={'tt': numpy.arange(start=yy_train.shape[0], stop=yy_train.shape[0]+yy_val.shape[0]), 'value': yy_val, 'kind': ['true'] * yy_val.shape[0], 'fold': ['val'] * yy_val.shape[0]})
c = pandas.DataFrame(data={'tt': numpy.arange(start=0, stop=yy_train_hat.shape[0]), 'value': yy_train_hat, 'kind': ['hat'] * yy_train_hat.shape[0], 'fold': ['train'] * yy_train_hat.shape[0]})
d = pandas.DataFrame(data={'tt': numpy.arange(start=yy_train_hat.shape[0], stop=yy_train_hat.shape[0]+yy_val_hat.shape[0]), 'value': yy_val_hat, 'kind': ['hat'] * yy_val_hat.shape[0], 'fold': ['val'] * yy_val_hat.shape[0]})
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

