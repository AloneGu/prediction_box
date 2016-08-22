import statsmodels.api as sm

import matplotlib.pyplot as plt

import pandas as pd

dta = pd.read_csv('../data/charge_test.csv')

idx = pd.DatetimeIndex(dta['stats_day'])

dta = pd.DataFrame(data=dta['charge_quantity'].values, index=idx,columns=['charge_quantity'])

print dta.head()

res = sm.tsa.seasonal_decompose(dta)

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(20, 10))
axes[0].plot(res.observed)
axes[0].set_ylabel('Observed')
axes[1].plot(res.trend)
axes[1].set_ylabel('Trend')
axes[2].plot(res.seasonal)
axes[2].set_ylabel('Seasonal')
axes[3].plot(res.resid)
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Time')
plt.show()
