#


#
import pandas


#


#
# data1 = pandas.read_csv('./raw/train_1.csv').set_index('Page').T
data = pandas.read_csv('./raw/train_2.csv').set_index('Page').T

data.to_csv('./dataset.csv', index=True)
