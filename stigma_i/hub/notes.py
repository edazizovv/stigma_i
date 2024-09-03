
raise Exception('hoho')
x_factors_proj = [x + '_ordinal' for x in ordinal_cols] + \
                 [x + '_onehot_{0}'.format(t)
                  for x in onehot_cols
                  for t in range(data[x_factors][x].value_counts().shape[0])] + \
                 nochange_cols
pandas.DataFrame(data=X_train, columns=x_factors_proj).to_csv('C:/Users/Edward/Desktop/x_train.csv', index=False)
pandas.DataFrame(data=X_test, columns=x_factors_proj).to_csv('C:/Users/Edward/Desktop/x_test.csv', index=False)
pandas.DataFrame(data=Y_train, columns=[target]).to_csv('C:/Users/Edward/Desktop/y_train.csv', index=False)
pandas.DataFrame(data=Y_test, columns=[target]).to_csv('C:/Users/Edward/Desktop/y_test.csv', index=False)
