import matplotlib.pyplot as plt
import matplotlib.finance as f
import numpy as np

spy = f.fetch_historical_yahoo('SPY', (2007,1,1), (2008,12,31))
agg = f.fetch_historical_yahoo('AGG', (2007,1,1,), (2008,12,31))

spy_l = f.parse_yahoo_historical(spy, adjusted=True, asobject=False)
agg_l = f.parse_yahoo_historical(agg, adjusted=True, asobject=False)
print spy_l
print agg_l

ndays = min(len(spy_l),len(agg_l)) - 1
print 'ndays', ndays

spy_daily = []
agg_daily = []
for i in range(1, ndays):
    spy_daily.append(float(spy_l[i][4]) / float(spy_l[i+1][4]) - 1)
    agg_daily.append(float(agg_l[i][4]) / float(agg_l[i+1][4]) - 1)
print spy_daily    
print agg_daily

x = np.array(spy_daily, dtype='float')
y = np.array(agg_daily, dtype='float')

plt.axis('equal')
plt.grid(True)
plt.xlabel('spy')
plt.ylabel('agg')
plt.title('daily returns during financial crisis')
plt.scatter(x,y, color='blue')
#plt.hist(x,rwidth=1)
plt.show()
