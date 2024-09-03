#


#
import pandas


#


#
data = pandas.read_csv('./raw/weatherHistory.csv')

data = data.drop(columns=['Loud Cover'])
data['Formatted Date'] = pandas.to_datetime(data['Formatted Date'])
data = data.rename(columns={'Formatted Date': 'date'})
data = data.set_index('date')

data.to_csv('./dataset.csv', index=True)
