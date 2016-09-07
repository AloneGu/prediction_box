import statsmodels.api as sm

import matplotlib.pyplot as plt

import pandas as pd

START = '2015-01-07'
MID = '2015-01-11'
END =  '2015-01-16'

dta = pd.read_csv('../data/nyc_taxi.csv')

dta['timestamp'] = pd.to_datetime(dta['timestamp'])

truth = dta[(dta.timestamp>START) & (dta.timestamp<END)]['value']

tmp_dta = dta[dta.timestamp<MID]

station_check = sm.tsa.stattools.adfuller(tmp_dta['value'].values)
print station_check
print round(station_check[1],4)

print tmp_dta.tail()

print 'acf',sm.tsa.acf(tmp_dta['value'].values)
print 'pacf',sm.tsa.pacf(tmp_dta['value'].values)

df = pd.Series(data=tmp_dta['value'].values,index=tmp_dta['timestamp'].values)

print df.head()

arma_mod = sm.tsa.ARMA(df, order=(3,1))
arma_res = arma_mod.fit(trend='nc', disp=-1)


print arma_res.summary()

fig, ax = plt.subplots(figsize=(10,8))
fig = arma_res.plot_predict(start=START, end=END, ax=ax)
legend = ax.legend(loc='upper left')
plt.show()

