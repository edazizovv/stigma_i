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


#
removables = ['New_Price']

target = 'Price'
x_factors = [x for x in data.columns if not any([y in x for y in [target] + removables])]


def treat_mileage(x):
    if 'km/kg' in x:
        return float(x[:x.index(' ')])
    elif 'kmpl' in x:
        return float(x[:x.index(' ')]) * 1.35
    else:
        return numpy.nan


data['Name'] = data['Name'].fillna(' ').apply(func=lambda x: x[:x.index(' ')])
data['Mileage'] = pandas.to_numeric(data['Mileage'].fillna(' ').apply(func=treat_mileage), errors='coerce')
data['Engine'] = pandas.to_numeric(data['Engine'].fillna(' ').apply(func=lambda x: x[:x.index(' ')]), errors='coerce')
data['Power'] = pandas.to_numeric(data['Power'].fillna(' ').apply(func=lambda x: x[:x.index(' ')]), errors='coerce')

data.loc[data['Power'].isna(), 'Power'] = data['Power'].median()
data.loc[data['Mileage'].isna(), 'Mileage'] = data['Mileage'].median()
data.loc[data['Engine'].isna(), 'Engine'] = data['Engine'].median()
data.loc[data['Seats'].isna(), 'Seats'] = data['Seats'].mode().values[0]

X = data[x_factors].values
Y = data[target].values


ordinal = OrdinalEncoder()
ordinal_cols = ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type', ]

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


class UsedCarPricePredictionNN(nn.Module):
    def __init__(self, n_cont, layers, out_sz, p=0.5):
        super().__init__()
        self.drpout = nn.Dropout(p)
        self.batchnorm = nn.BatchNorm1d(n_cont)

        layerslist = []
        n_in = n_cont

        for i in layers:
            layerslist.append(nn.Linear(n_in, i))
            layerslist.append(nn.ReLU(inplace=True))
            layerslist.append(nn.BatchNorm1d(i))
            layerslist.append(nn.Dropout(p))
            n_in = i
        layerslist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerslist)

    def forward(self, x_cont):
        x_cont = self.batchnorm(x_cont)
        x = x_cont
        x = self.layers(x)
        return x

model = UsedCarPricePredictionNN(X_train_.shape[1], [100, 50], 1, p = 0.4)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)
epochs=2400
final_losses=[]
for i in range(epochs):
    y_pred=model(X_train_)
    loss=torch.sqrt(loss_function(y_pred,Y_train_)) ### RMSE
    final_losses.append(loss)
    if i%100==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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