import statsmodels.api as sm

import matplotlib.pyplot as plt
dta = sm.datasets.co2.load_pandas().data

# deal with missing values. see issue
dta.co2.interpolate(inplace=True)

print dta.head()

res = sm.tsa.seasonal_decompose(dta)

fig, axes = plt.subplots(4, 1, sharex=True,figsize=(20,10))
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