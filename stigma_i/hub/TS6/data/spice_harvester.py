#


#
import pandas


#


#
data = pandas.read_csv('./raw/household_power_consumption.txt', sep=';', na_values='?')

data['date'] = pandas.to_datetime(data['Date'] + ' ' + data['Time'])
data['meter'] = data['Global_active_power'] * 1000 / 60 - data['Sub_metering_1'] - data['Sub_metering_2'] - data['Sub_metering_3']
data = data.drop(columns=['Date', 'Time']).set_index('date')

data.to_csv('./dataset.csv', index=True)
