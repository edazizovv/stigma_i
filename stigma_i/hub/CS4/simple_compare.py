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
data = data.set_index('BBL')


#
random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999


#
data['BUILDING CLASS CATEGORY'] = data['BUILDING CLASS CATEGORY'].str.split().str[1]
data['TAX_CLASS_AT_PRESENT_P1'] = data['TAX CLASS AT PRESENT'].str[0].fillna('O')
data['TAX_CLASS_AT_PRESENT_P2'] = data['TAX CLASS AT PRESENT'].str[1].fillna('O')
data['BUILDING_CLASS_AT_PRESENT_P1'] = data['BUILDING CLASS AT PRESENT'].str[0].fillna('O')
data['BUILDING_CLASS_AT_PRESENT_P2'] = data['BUILDING CLASS AT PRESENT'].str[1].fillna('O')

data['LAND SQUARE FEET'] = pandas.to_numeric(data['LAND SQUARE FEET'], errors='coerce')
data['LAND SQUARE FEET'] = data['LAND SQUARE FEET'].fillna(data['LAND SQUARE FEET'].median())
data['GROSS SQUARE FEET'] = pandas.to_numeric(data['GROSS SQUARE FEET'], errors='coerce')
data['GROSS SQUARE FEET'] = data['GROSS SQUARE FEET'].fillna(data['GROSS SQUARE FEET'].median())
data.loc[data['YEAR BUILT'] == 0, 'YEAR BUILT'] = numpy.nan
data['YEAR BUILT'] = data['YEAR BUILT'].fillna(data['YEAR BUILT'].median())

data['BUILDING_CLASS_AT_TIME_OF_SALE_P1'] = data['BUILDING CLASS AT TIME OF SALE'].str[0].fillna('O')
data['BUILDING_CLASS_AT_TIME_OF_SALE_P2'] = data['BUILDING CLASS AT TIME OF SALE'].str[1].fillna('O')

data['SALE_DATE_YEAR'] = pandas.to_datetime(data['SALE DATE']).dt.year
data['SALE_DATE_MONTH'] = pandas.to_datetime(data['SALE DATE']).dt.month
data['SALE_DATE_DAY'] = pandas.to_datetime(data['SALE DATE']).dt.day

data['SALE PRICE'] = pandas.to_numeric(data['SALE PRICE'], errors='coerce')

#
removables = ['BLOCK', 'LOT', 'NEIGHBORHOOD', 'EASE-MENT', 'ADDRESS', 'APARTMENT NUMBER', 'ZIP CODE',
              'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE',
              'SALE DATE']

target = 'SALE PRICE'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

data = data[[target] + x_factors].dropna()
thresh = 10_000
data = data[data[target] > thresh]

X = data[x_factors].values
Y = data[target].values


ordinal = OrdinalEncoder()
ordinal_cols = ['TAX_CLASS_AT_PRESENT_P1', 'TAX_CLASS_AT_PRESENT_P2',
                'BUILDING_CLASS_AT_PRESENT_P1', 'BUILDING_CLASS_AT_PRESENT_P2',
                'BUILDING_CLASS_AT_TIME_OF_SALE_P1', 'BUILDING_CLASS_AT_TIME_OF_SALE_P2']

ordinal.fit(X=X[:, [x_factors.index(x) for x in ordinal_cols]])

onehot = OneHotEncoder()
onehot_cols = ['BUILDING CLASS CATEGORY']
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

# bench = Y_train.mean()
bench = numpy.median(Y_train)

y_hat_train = numpy.full(shape=Y_train_.shape, fill_value=bench)
y_hat_test = numpy.full(shape=Y_test_.shape, fill_value=bench)

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

"""