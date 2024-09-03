#


#
import pandas


#


#
data = pandas.read_csv('./raw/D202.csv')

data['date'] = pandas.to_datetime(data['DATE'] + ' ' + data['START TIME'])
data = data[['date', 'USAGE']].copy()
data = data.set_index('date')

data.to_csv('./dataset.csv', index=True)
