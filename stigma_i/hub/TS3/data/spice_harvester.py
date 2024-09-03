#
import os


#
import calendar
import numpy
import pandas


#


#
drops = ['D-J-F', 'M-A-M', 'J-J-A', 'S-O-N', 'metANN']
month_codes = {month.lower(): index for index, month in enumerate(calendar.month_abbr) if month}

data = []
for f in os.listdir('./raw'):
    sliced = pandas.read_csv('./raw/{0}'.format(f))
    sliced = sliced.drop(columns=drops)
    sliced = sliced.set_index('YEAR')
    sliced = sliced.stack().reset_index()
    sliced = sliced.rename(columns={'level_1': 'MONTH', 0: 'T_{0}'.format(f)})
    sliced['date'] = pandas.to_datetime(pandas.DataFrame(
        data={'YEAR': sliced['YEAR'].values,
              'MONTH': sliced['MONTH'].apply(func=lambda x: month_codes[x.lower()]).values,
              'DAY': numpy.ones(shape=(sliced.shape[0],))}))
    sliced = sliced.drop(columns=['YEAR', 'MONTH'])
    sliced = sliced.set_index('date')
    data.append(sliced)
data = pandas.concat(data, axis=1).reset_index()

data.to_csv('./dataset.csv', index=False)
