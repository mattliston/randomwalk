from __future__ import division
import argparse
import numpy as np
import h5py
import caffe
from scipy.cluster.vq import kmeans,vq, kmeans2, whiten

# MAIN PROGRAM
parser = argparse.ArgumentParser()
parser.add_argument('--window', help='window size in trading days')
parser.add_argument('--test_split', help='fraction of dates used for clustering (from most recent), 0.0 to 1.0')
parser.add_argument('--k', help='number of kmeans clusters')
parser.add_argument('--whiten', help='normalize feature vectors before clustering')
parser.add_argument('--minit', help='kmeans initialization method, random or points')
args = parser.parse_args()
print args.window, args.test_split, args.k, args.whiten
window = int(args.window)
test_split = float(args.test_split)
k = int(args.k)

# load raw prices
h5 = h5py.File("nyse_nasdaq.hdf5", "r")
prices = np.array(h5.get('close'))
prices = np.fliplr(prices) # reverse order to be oldest->newest
symbols = np.array(h5.get('symbols'))
h5.close()
print 'prices.shape', prices.shape

# compute daily changes
changes = np.zeros((prices.shape[0], prices.shape[1] - 1), dtype=np.float32)
changes = (prices[:,1:] / prices[:,0:-1]) - 1.0
nsyms = changes.shape[0]
ndays = changes.shape[1]
print 'changes.shape', changes.shape

# load trained network
caffe.set_mode_gpu()
net = caffe.Net('deploy.prototxt', 'snapshot_iter_10000000.caffemodel', caffe.TEST)

date_split = int(ndays*float(test_split))
print 'date_split', date_split

print 'computing feature vectors for training set'
fa = [] # feature vector output from network
ta = [] # probability output from network
ca = [] # changes
cn = [] # next day change
sa = [] # symbol index
pa = [] # prices
for i in range(nsyms):
    for j in range(0,ndays-date_split-window-1): # training date range, oldest to newest
        net.blobs['data'].data[0,:,0,0] = changes[i,j:j+window]
        net.forward()
        feat = net.blobs['fc6'].data[0]
        prob = net.blobs['prob'].data[0]
        fa.append(feat.copy())
        ta.append(prob.copy()[0])
        ca.append(changes[i,j+window-1].copy())
        cn.append(changes[i,j+window].copy())
        sa.append(i)
        pa.append(prices[i,j+window].copy())

f = np.array(fa, dtype=np.float32)
print 'training f.shape', f.shape
w = whiten(f)

if (args.whiten):
    print 'running kmeans on whitened features'
    centroids,label = kmeans2(w, k, minit=args.minit)
else:
    print 'running kmeans on raw features'
    centroids,label = kmeans2(f, k, minit=args.minit)
print 'centroids.shape', centroids.shape, 'label.shape', label.shape

print 'computing feature vectors for testing set'
fa = [] # feature vector output from network
ta = [] # probability output from network
ca = [] # changes
cn = [] # next day change
sa = [] # symbol index
pa = [] # prices
for i in range(nsyms):
    for j in range(ndays-date_split, ndays-window-1):
        net.blobs['data'].data[0,:,0,0] = changes[i,j:j+window]
        net.forward()
        feat = net.blobs['fc6'].data[0]
        prob = net.blobs['prob'].data[0]
        fa.append(feat.copy()) # feature array
        ta.append(prob.copy()[0]) # typicality array
        ca.append(changes[i,j+window-1].copy()) # change array
        cn.append(changes[i,j+window].copy()) # next day change array
        sa.append(i) # symbol array
        pa.append(prices[i,j+window].copy()) # price array

f = np.array(fa, dtype=np.float32)
print 'testing f.shape', f.shape
w = whiten(f)

if (args.whiten):
    print 'running vq on whitened features'
    code, dist = vq(w, centroids)
else:
    print 'running vq on raw features'
    code, dist = vq(f, centroids)
print 'code.shape', code.shape, 'dist.shape', dist.shape

print 'generating cluster_detail.csv'
fcsv=open('./cluster_detail.csv', 'w')
print >>fcsv, 'symbol_index symbol_name cluster_number prob price change next_change'
m = [[] for i in range(k)] # create empty array per cluster to hold next day changes, so we can compute the mean and variance
for i in range(len(code)):
    print >>fcsv, sa[i], symbols[sa[i]], code[i], ta[i], pa[i], ca[i], cn[i]
    if ta[i] > 0.9:
        m[code[i]].append(cn[i])
fcsv.close()

print 'generating total market array for test period'
mt = []
for i in range(nsyms):
    for j in range(ndays-date_split, ndays):
        mt.append(changes[i,j])

print 'generating cluster_summary.csv'
fcsv=open('./cluster_summary.csv', 'w')
print >>fcsv, 'cluster_number n mean std up'
print >>fcsv, 'market_test', len(mt), np.mean(np.array(mt)), np.std(np.array(mt)), np.sign(np.array(mt)).sum() / len(mt)
for i in range(k):
    if len(m[i]) > 0:
        print >>fcsv, i, len(m[i]), np.mean(np.array(m[i])), np.std(np.array(m[i])), np.sign(np.array(m[i])).sum() / len(m[i])
fcsv.close()

#END OF PROGRAM
