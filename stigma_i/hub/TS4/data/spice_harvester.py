#


#
import pandas


#


#
data = pandas.read_csv('./raw/POP.csv')
data = data.drop(columns=['realtime_start', 'realtime_end']).set_index('date').rename(columns={'value': 'population'})
data.index = pandas.to_datetime(data.index)

data.to_csv('./dataset.csv', index=True)
