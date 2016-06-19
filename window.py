import argparse
import numpy as np
import h5py

def create_dataset(changes, window, shuffle_both, prefix):
    nsyms = changes.shape[0]
    ndays = changes.shape[1]
    data = np.zeros((nsyms * (ndays - window) * 2, window), dtype=np.float32)
    label = np.zeros((nsyms * (ndays - window) * 2), dtype=np.int32)

    print 'create_dataset', changes.shape, window, shuffle_both, prefix
    k = 0
    for i in range(nsyms):
        for j in range(ndays - window):
            if shuffle_both:
                data[k,:] = np.random.permutation(changes[i, j:j+window])
            else:
                data[k,:] = changes[i, j:j+window]
            label[k] = 0
            k = k + 1
            data[k,:] = np.random.permutation(changes[i, j:j+window])
            label[k] = 1
            k = k + 1

    # shuffle data and label in unison
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(label)

    # split hdf data into n files of size s, due to 2GB limit in caffe
    n = 5
    s = int(data.shape[0] * (1.0/float(n)))
    flist=open(prefix+'.flist', 'w+')
    for i in range(n):
        print >>flist, prefix+str(i)+'.hdf5'
        h5 = h5py.File(prefix+str(i)+'.hdf5', 'w')
        h5.create_dataset('data', data=np.array(data[(i)*s:(i+1)*s,:]), dtype='f')
        h5.create_dataset('label', data=np.array(label[(i)*s:(i+1)*s]), dtype='i')
        h5.close()
    flist.close()

# MAIN PROGRAM
parser = argparse.ArgumentParser()
parser.add_argument('--window', help='window size in trading days')
parser.add_argument('--shuffle_both', help='shuffle label=0 and label=1')
parser.add_argument('--test_split', help='fraction of dates used for testing (from most recent), 0.0 to 1.0')
parser.add_argument('--quantize', help='quantize closing price changes to -1,0,+1')
args = parser.parse_args()
print args.window, args.shuffle_both, args.test_split
window = int(args.window)

# load raw prices
h5 = h5py.File("nyse_nasdaq.hdf5", "r")
prices = np.array(h5.get('close'))
prices = np.fliplr(prices) # order from oldest date to newest date
print 'prices.shape', prices.shape
h5.close()

# compute daily changes
changes = np.zeros((prices.shape[0], prices.shape[1] - 1), dtype=np.float32)
changes = (prices[:,1:] / prices[:,0:-1]) - 1.0 # (close / previous_close) - 1
np.random.shuffle(changes) # shuffle by symbol
print 'changes.shape', changes.shape
if args.quantize:
    print 'quantizing to -1,0,+1'
    changes = np.sign(changes) # quantize to -1,0,+1

date_split = int(changes.shape[1]*float(args.test_split))
create_dataset(changes[:,-date_split:], window, args.shuffle_both, 'test')
create_dataset(changes[:,0:-date_split], window, args.shuffle_both, 'train')
