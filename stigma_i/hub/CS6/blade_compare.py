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
from tests import check_up


#
data = pandas.read_csv('./data/dataset.csv')


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#
cities = data['ADDRESS'].str.split(',').str[1]
cities_chosen = cities.value_counts()[cities.value_counts() >= 1000].index
data['ADDRESS'] = cities.apply(func=lambda x: x if x in cities_chosen else 'OTHER')

#
removables = ['BHK_OR_RK']


target = 'TARGET(PRICE_IN_LACS)'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

data = data[[target] + x_factors].dropna()

X = data[x_factors].values
Y = data[target].values


ordinal = OrdinalEncoder()
ordinal_cols = ['POSTED_BY']

ordinal.fit(X=X[:, [x_factors.index(x) for x in ordinal_cols]])

onehot = OneHotEncoder()
onehot_cols = ['ADDRESS']
onehot.fit(X=X[:, [x_factors.index(x) for x in onehot_cols]])

nochange_cols = [x for x in x_factors if x not in ordinal_cols + onehot_cols]

X = numpy.concatenate((ordinal.transform(X=X[:, [x_factors.index(x) for x in ordinal_cols]]),
                       onehot.transform(X=X[:, [x_factors.index(x) for x in onehot_cols]]).toarray(),
                       X[:, [x_factors.index(x) for x in nochange_cols]]), axis=1).astype(dtype=float)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=rs)

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

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import Dropout

model_price = Sequential()

# Number of neurons equal to te feautres on the dataset
model_price.add(Dense(8,activation='relu',input_shape=(X_train_.shape[1],)))
model_price.add(Dropout(0.5))
model_price.add(Dense(8,activation='relu'))
model_price.add(Dropout(0.5))
model_price.add(Dense(8,activation='relu'))
model_price.add(Dropout(0.5))
model_price.add(Dense(8,activation='relu'))
model_price.add(Dropout(0.5))
model_price.add(Dense(1, activation = 'linear'))

model_price.compile(optimizer='adam',loss='mae')

from tensorflow.keras.callbacks import EarlyStopping
cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model_price.fit(x=X_train_,y=Y_train_, validation_data=(X_test, Y_test), batch_size=128, epochs=150, callbacks=[cb])
"""
# """
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mae
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.activations import linear, relu

class stopLearn(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_loss') < 21:
            print("\nStop training!!")
            self.model.stop_training=True

cb = stopLearn()
# checkpoint_name = 'models/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
# cp = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [cp, cb]

model = Sequential()
model.add(Dense(units=32, kernel_initializer='normal', input_dim=X_train_.shape[1], activation=relu))   # ЖЕСТЬ ЭФФЕКТ ОТ КЕРНЕЛ ИНИТ!
model.add(Dense(units=64, activation=relu))
model.add(Dense(units=64, activation=relu))
model.add(Dense(units=128, activation=relu))
model.add(Dense(units=256, activation=relu))
model.add(Dense(units=1, activation=linear))

model.compile(optimizer=Adam(learning_rate=0.001), loss=mae)
# model.compile(optimizer=SGD(learning_rate=0.001), loss=mae)

model.summary()

model.fit(x=X_train_, y=Y_train_, validation_data=(X_train_, Y_train_), batch_size=128, epochs=500, callbacks=[cb])

# """

y_hat_train = model.predict(X_train_)
y_hat_test = model.predict(X_test_)

results_train = check_up(Y_train.flatten(), y_hat_train.flatten(), None, X_train_)
results_test = check_up(Y_test.flatten(), y_hat_test.flatten(), None, X_test_)

results_train['sample'] = 'train'
results_test['sample'] = 'test'

results_train = pandas.DataFrame(pandas.Series(results_train))
results_test = pandas.DataFrame(pandas.Series(results_test))

"""
# joblib.dump(model, filename='./model_ex12.pkl')
results_train.T.to_csv('./reported.csv', mode='a', header=False)
results_test.T.to_csv('./reported.csv', mode='a', header=False)

fig, ax = pyplot.subplots(2, 2, sharex='col', sharey='col')
ax[0, 0].plot(range(Y_train.shape[0]), Y_train.flatten() - y_hat_train.flatten(), color='navy')
ax[1, 0].plot(range(Y_test.shape[0]), Y_test.flatten() - y_hat_test.flatten(), color='orange')
ax[0, 1].hist(Y_train.flatten() - y_hat_train.flatten(), color='navy', bins=100, density=True)
ax[1, 1].hist(Y_test.flatten() - y_hat_test.flatten(), color='orange', bins=100, density=True)


fig, ax = pyplot.subplots(2, 2, sharex='col', sharey='col')
ax[0, 0].plot(range(Y_train[:100].shape[0]), Y_train.flatten()[:100] - y_hat_train.flatten()[:100], color='navy')
ax[1, 0].plot(range(Y_test[:100].shape[0]), Y_test.flatten()[:100] - y_hat_test.flatten()[:100], color='orange')
ax[0, 1].hist(Y_train.flatten()[:100] - y_hat_train.flatten()[:100], color='navy', bins=100, density=True)
ax[1, 1].hist(Y_test.flatten()[:100] - y_hat_test.flatten()[:100], color='orange', bins=100, density=True)


"""