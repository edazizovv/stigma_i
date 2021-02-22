
import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as R2_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from IPython.display import display_html

import os


data_train = pd.read_csv('./data/DailyDelhiClimateTrain.csv',sep=',')
data_test = pd.read_csv('./data/DailyDelhiClimateTest.csv',sep=',')
data = pd.concat([data_train,data_test])

data['date'] = pd.to_datetime(data['date'])
data = data.rename(columns={"meantemp":"temp","wind_speed":"wind","meanpressure":"pressure"})


dates = data['date'].values
temp  = data['temp'].values
humidity = data['humidity'].values
wind = data['wind'].values
pressure = data['pressure'].values


eps = 1e-11


# Logarithmic transformation

def transform_logratios(serie):
    aux = np.log((serie[1:]+eps) / (serie[0:-1]+eps))
    return np.hstack( ([np.nan], aux))
def inverse_transform_logratios(log_ratio, temp_prev):
    return np.multiply(temp_prev, np.exp(log_ratio))

transform = transform_logratios
inverse_transform = inverse_transform_logratios

scaler = MinMaxScaler()
transformed = scaler.fit_transform(data.loc[:, ["humidity","wind","pressure"]])

transformed = pd.DataFrame(transformed, columns = ["humidity_s","wind_s","pressure_s"])
transformed.head()

humidity_s = transformed['humidity_s'].values
wind_s = transformed['wind_s'].values
pressure_s = transformed['pressure_s'].values

def winnowing(series, target, prev_known,
              W_in=1, W_out=1):
    n = len(series[0])
    dataX = np.nan *np.ones((n ,W_in ,len(series)))
    if np.sometrue([s.dtype == object for s in series]):
        dataX = dataX.astype(object)
    if W_out==1:
        dataY = series[target].copy()
    else:
        dataY = np.nan *np.ones((n ,W_out))
        if series[target].dtype == object:
            dataY = dataY.astype(object)
        dataY[: ,0] = series[target].copy()
        for i in range(1 ,W_out):
            dataY[:-i ,i] = dataY[i: ,0].copy()

    for i in range(n):
        for j ,s in enumerate(prev_known):
            int_s = int(s)
            ini_X = max([0 ,W_in - i -int_s])
            dataX[i, ini_X: ,j] = \
                series[j][max([0 , i -W_in +int_s]):min([n , i +int_s])]

    return dataX, dataY


def my_dfs_display(dfs,names):
    df_styler = []
    for df,n in zip(dfs,names):
        df_styler.append(df.style.set_table_attributes("style='display:inline'").\
                         set_caption(n))
    display_html(df_styler[0]._repr_html_()+"__"+df_styler[1]._repr_html_(),
                 raw=True)


def info_winnowing(X,Y,names_series,name_target,times=None):
    c0  = '\033[1m'
    c1  = '\033[0m'
    W_in = X.shape[1]
    if len(Y.shape)==1:
        W_out = 1
    else:
        W_out = Y.shape[1]
    print(len(X), "windows created \n")
    print("X.shape={}".format(X.shape)," Y.shape={}".format(Y.shape),"\n")
    for t in range(len(X)):
        print(c0,"Window %d:"%t, c1)
        if times is None:
            names_ts = ["t="+str(t+i-W_in) for i in range(W_in)]
            names_ts_pred = ["t="+str(t+i) for i in range(W_out)]
        else:
            times = list(times)
            if (t-W_in)<0:
                names_ts = ["?"+str(i) for i in range(W_in-t)] + times[:t]
            else:
                names_ts = times[(t-W_in):t]
            if (t+W_out-1)>=len(times):
                names_ts_pred = times[t:] + ["?"+str(i) for i in range(W_out-(len(times)-t))]
            else:
                names_ts_pred = times[t:(t+W_out)]
        aux1 = pd.DataFrame(X[t].T,columns=names_ts,index=names_series)
        aux2 = pd.DataFrame([Y[t]],columns=names_ts_pred,
                            index=[name_target])
        if W_out==1:
            my_dfs_display((aux1,aux2),
                           ("X[{}].shape={}".format(t,X[t].shape),
                            "Y[{}]={}".format(t,Y[t])))
        else:
            my_dfs_display((aux1,aux2),
                           ("X[{}].shape={}".format(t,X[t].shape),
                            "Y[{}].shape={}".format(t,Y[t].shape)))


logratio_temp = transform(temp)


series = [logratio_temp, humidity_s, wind_s, pressure_s]
# series = [temp, humidity, wind, pressure]
prev_known = [False, False, False, False]

lookback = 6  # Window_in

X, y = winnowing (series, target=0, prev_known=prev_known,
                  W_in=lookback)

print(X.shape, np.shape(y))

X_train = X[(lookback+1):len(data_train)]
y_train = y[(lookback+1):len(data_train)]
temp_train = temp[(lookback+1):len(data_train)]
temp_test  = temp[len(data_train):]
X_test  = X[len(data_train):]
y_test  = y[len(data_train):]

print(np.shape(temp_train))
print(np.shape(temp_test))

temp_prev_train =  np.hstack(( [np.nan], temp_train[:-1]))
temp_prev_test  =  np.hstack(( temp_train[-1:],
                                      temp_test[:-1]))
dates_train     = dates[(lookback+1):len(data_train)]
dates_test      = dates[len(data_train):]

model = Sequential()
model.add(LSTM(10, input_shape=(lookback, X_train.shape[2]),
#              kernel_regularizer='l1'
              )
         )
model.add(Dense(1,
#                kernel_regularizer='l1'
               )
         )
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

def training_graphic(tr_mse, val_mse):
    ax=plt.figure(figsize=(10,4)).gca()
    plt.plot(1+np.arange(len(tr_mse)), tr_mse)
    plt.plot(1+np.arange(len(val_mse)), val_mse)
    plt.title('mse', fontsize=18)
    plt.xlabel('time', fontsize=18)
    plt.ylabel('mse', fontsize=18)
    plt.legend(['Training', 'Validation'], loc='upper left')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


epochs = 200
batch_size = 64
Nval = 200
control_val = True
save_training_tensorboard = False

callbacks = Callback()

if not control_val:
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, verbose=2)

else:
    acum_tr_mse = []
    acum_val_mse = []
    filepath = "./best_model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=2,
                                 save_best_only=True,
                                 mode='min')

    if save_training_tensorboard:
        callbacks_list = callbacks + [checkpoint]
    else:
        callbacks_list = [checkpoint]

    for e in range(epochs):
        history = model.fit(X_train[:-Nval], y_train[:-Nval],
                            batch_size=batch_size,
                            epochs=1,
                            callbacks=callbacks_list,
                            verbose=0,
                            validation_data=(X_train[-Nval:], y_train[-Nval:]))

        acum_tr_mse += history.history['mse']
        acum_val_mse += history.history['val_mse']

        if (e + 1) % 50 == 0:
            training_graphic(acum_tr_mse, acum_val_mse)

model = load_model('./best_model.h5')


y_train_prediction = model.predict(X_train).flatten()
y_test_prediction = model.predict(X_test).flatten()



temp_train_pred = inverse_transform(y_train_prediction,
                                          temp_prev_train)
temp_test_pred  = inverse_transform(y_test_prediction,
                                          temp_prev_test)



plt.figure(figsize=(15,7))
plt.plot(dates_train, temp_train, '--', c='royalblue',
         label="Training")
plt.plot(dates_train, temp_train_pred,  c='darkorange',
         label="Training daily predictions")

plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.axis([dates_train[4],dates_train[-1],0,75])
plt.legend(fontsize=14)



plt.figure(figsize=(15,5))
plt.plot(dates_train, temp_train, '--', c='royalblue',
         label='Training')
plt.plot(dates_train, temp_train_pred,  c='darkorange',
         label='Training predictions')
plt.plot(dates_test, temp_test, '--',   c='green',
         label='Test')
plt.plot(dates_test, temp_test_pred,    c='red',
         label='Test predictions')
plt.title('Daily predictions (zoom)', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.axis([dates_train[-200],dates_test[-100],0,50])


# R2 scores
print("R2 - Training      : ",
      R2_score(temp_train[1:], temp_train_pred[1:]))
print("R2 - Test          : ",
      R2_score(temp_test, temp_test_pred))
print("r2 - Interval 1 day     : ",
      R2_score(temp_test[1:], temp_test[:-1]))
print("R2 - Interval 1 week : ",
      R2_score(temp_test[7:], temp_test[:-7]))
print("R2 - Interval 4 weeks: ",
      R2_score(temp_test[28:], temp_test[:-28]))
print("R2 - Interval 1 year: ",
      R2_score(temp_train[7*52:], temp_train[:-7*52]))

# RMSEs
sqrt = np.sqrt
print("RMSE - Training      : ",
      sqrt(mean_squared_error(temp_train[1:],
                              temp_train_pred[1:])))
print("RMSE - Test          : ",
      sqrt(mean_squared_error(temp_test,
                              temp_test_pred)))
print("RMSE - Interval 1 day    : ",
      sqrt(mean_squared_error(temp_test[1:],
                              temp_test[:-1])))
print("RMSE - Interval 1 week : ",
      sqrt(mean_squared_error(temp_test[7:],
                              temp_test[:-7])))
print("RMSE - Interval 4 weeks: ",
      sqrt(mean_squared_error(temp_test[28:],
                              temp_test[:-28])))
