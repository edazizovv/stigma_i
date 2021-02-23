#


#
import pandas


#


#
d = './data/ma_lga_12345.csv'
data = pandas.read_csv(d)

data = data.rename(columns={'saledate': 'date'})
data = data.rename(columns={'MA': 'ma'})
data['date'] = pandas.to_datetime(data['date'])
data = data.set_index('date')

# currently we do not accept categorical features
data = data.drop(columns='type')

data[['{0}_LAG1'.format(x) for x in data.columns.values]] = data[[x for x in data.columns.values]].shift(1)
data = data.iloc[1:, :].copy()

thresh = 300

target = 'ma'
factors = [x for x in data.columns.values if 'LAG' not in x]
x_train = data[factors].iloc[:thresh, :]
y_train = data[target].iloc[:thresh, :]
x_val = data[factors].iloc[thresh:, :]
y_val = data[target].iloc[thresh:, :]
