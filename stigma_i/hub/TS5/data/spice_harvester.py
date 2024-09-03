#


#
import pandas


#


#
data = pandas.read_csv('./raw/MLTollsStackOverflow.csv')
data = data.rename(columns={'month': 'date'})
data = data.set_index('date').drop(columns=['Tableau'])

data.to_csv('./dataset.csv', index=True)
