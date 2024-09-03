#
import os


#
import pandas


#


#
d = './hub/{0}/{1}'

reported, results = [], []
for j in range(8):
    ds = 'CS{0}'.format(j+1)

    dd = d.format(ds, 'reported.csv')
    reported_slice = pandas.read_csv(dd)
    reported_slice = reported_slice.rename(columns={reported_slice.columns[0]: 'ex'})
    reported_slice['dataset'] = ds
    reported.append(reported_slice)

    dh = d.format(ds, 'results.csv')
    results_slice = pandas.read_csv(dh)
    results_slice = results_slice.rename(columns={results_slice.columns[0]: 'ex'})
    results_slice['dataset'] = ds
    results.append(results_slice)

reported = pandas.concat(reported, axis=0, ignore_index=True)
results = pandas.concat(results, axis=0, ignore_index=True)

joint = reported.merge(right=results, left_on=['dataset', 'ex'], right_on=['dataset', 'ex'], how='outer')
