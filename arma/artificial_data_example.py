
## Autoregressive Moving Average (ARMA): Artificial data


import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(12345)


# Generate some data from an ARMA process:

arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])


# The conventions of the arma_generate function require that we specify a 1 for the zero-lag of the AR and MA parameters and that the AR parameters be negated.

arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]
nobs = 250
y = arma_generate_sample(arparams, maparams, nobs)


#  Now, optionally, we can add some dates information. For this example, we'll use a pandas time series.

import pandas as pd
dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
print dates[-10:]

station_check = sm.tsa.stattools.adfuller(y)
print station_check
print round(station_check[1],4)

y = pd.TimeSeries(y, index=dates)
arma_mod = sm.tsa.ARMA(y, order=(2,2))
arma_res = arma_mod.fit(trend='nc', disp=-1)

print arma_res.summary()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,8))
fig = arma_res.plot_predict(start='2000m1', end='2001m5', ax=ax)
legend = ax.legend(loc='upper left')
plt.show()