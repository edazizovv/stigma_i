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

data['floor'] = pandas.to_numeric(data['floor'], errors='coerce')
data['floor'] = data['floor'].fillna(0)

#
removables = ['hoa (R$)', 'property tax (R$)', 'fire insurance (R$)', 'total (R$)']


target = 'rent amount (R$)'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]

data = data[[target] + x_factors].dropna()

X = data[x_factors].values
Y = data[target].values


ordinal = OrdinalEncoder()
ordinal_cols = ['city', 'animal', 'furniture']

ordinal.fit(X=X[:, [x_factors.index(x) for x in ordinal_cols]])

nochange_cols = [x for x in x_factors if x not in ordinal_cols]

X = numpy.concatenate((ordinal.transform(X=X[:, [x_factors.index(x) for x in ordinal_cols]]),
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

X_train_ = torch.tensor(X_train_, dtype=torch.float)
Y_train_ = torch.tensor(Y_train_, dtype=torch.float)
X_test_ = torch.tensor(X_test_, dtype=torch.float)
Y_test_ = torch.tensor(Y_test_, dtype=torch.float)


# create class
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(LinearRegression, self).__init__()
        # Linear function.
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):  # x:inputs
        return self.linear(x)


# define model
input_dim = X_train_.shape[1]
output_dim = 1
model = LinearRegression(input_dim, output_dim)  #

# MSE
mse = nn.MSELoss()

# Optimization (find parameters that minimize error)
learning_rate = 0.02  # how fast we reach best parameters
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train model
loss_list = []
iteration_number = 500
for iteration in range(iteration_number):

    # optimization
    optimizer.zero_grad()

    # Forward to get output
    results = model(X_train_)

    # Calculate Loss
    loss = mse(results, Y_train_)

    # backward propagation
    loss.backward()

    # Updating parameters
    optimizer.step()

    # store loss
    loss_list.append(loss.data)

    # print loss
    if (iteration % 50 == 0):
        print('epoch {}, loss {}'.format(iteration, loss.data))

y_hat_train = model(X_train_)
y_hat_test = model(X_test_)

results_train = check_up(Y_train.flatten(), y_hat_train.detach().numpy().flatten(), None, X_train_)
results_test = check_up(Y_test.flatten(), y_hat_test.detach().numpy().flatten(), None, X_test_)

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