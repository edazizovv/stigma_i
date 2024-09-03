#


#
import pandas


#


#
data1 = pandas.read_csv('./raw/DailyDelhiClimateTest.csv').iloc[1:, :]
data2 = pandas.read_csv('./raw/DailyDelhiClimateTrain.csv')

data = pandas.concat((data2, data1), axis=0, ignore_index=True)

data = data.set_index('date')

data.to_csv('./dataset.csv', index=False)
