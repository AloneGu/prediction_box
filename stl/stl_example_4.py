import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import datetime

dta = pd.read_csv('../data/c89346c24315_rate.csv')

v = dta['display_temp'].values
t = dta['date'].values
res_dict = {}
for tmp_t,tmp_v in zip(t,v):
    tmp_t_str = str(tmp_t).strip()
    tmp_t_obj = datetime.datetime.strptime(tmp_t_str,'%Y/%m/%d %H:%M:%S')
    tmp_t_obj -= datetime.timedelta(seconds=tmp_t_obj.second)
    res_dict[tmp_t_obj] = tmp_v

df = pd.DataFrame.from_dict(res_dict,orient='index')
df.columns = ['value']


df = df.sort()

idx = pd.date_range(start=df.index[0],end=datetime.datetime(2016,6,15),freq='H')

df = df.reindex(idx).interpolate()

print df


res = sm.tsa.seasonal_decompose(df.values,freq=24)

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
