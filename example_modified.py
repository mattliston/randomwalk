import urllib2
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
print 'numpy ' + np.__version__

spy = urllib2.urlopen('http://real-chart.finance.yahoo.com/table.csv?s=SPY').read().splitlines()
print spy
agg = urllib2.urlopen('http://real-chart.finance.yahoo.com/table.csv?s=AGG').read().splitlines()

horizon = 1 # in trading days
ndays = min(len(spy),len(agg)) - horizon
print 'ndays', ndays

spy_r=[]
agg_r=[]
for i in range(1, ndays):
    # Date,Open,High,Low,Close,Volume,Adj Close
    if spy[i].split(',')[0] != agg[i].split(',')[0]:
        print 'error: date mismatch', spy[i].split(',')[0], agg[i].split(',')[0]
        exit(0)
    spy_r.append(float(spy[i].split(',')[4]) / float(spy[i+horizon].split(',')[4]) - 1)
    agg_r.append(float(agg[i].split(',')[4]) / float(agg[i+horizon].split(',')[4]) - 1)

x = np.array(spy_r, dtype='float')
y = np.array(agg_r, dtype='float')
print 'x.shape', x.shape, 'y.shape', y.shape

mean=[]
p_value=[]
for j in range(1, 10000): #compute sample means and p-value
    sample = np.array(random.sample(x,30), dtype='float')
    sample_mean = np.mean(sample)
    mean.append(sample_mean)
    p_value.append(stats.shapiro(sample)[1])

print mean
print p_value

m = np.array(mean,'float')
p = np.array(p_value,'float')

plt.axis([min(m), max(m), 0, 1])
plt.grid(True)
plt.xlabel('mean')
plt.ylabel('p-value')
plt.scatter(m, p, color='blue')
plt.show()
