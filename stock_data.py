#mnist_data.py
"""

This file is used to load and process stock data from iextrading, 
and is imported

It uses a moving frame to calculate
"""
import pandas as pd
import numpy as np
from scipy import random, linalg


print('==>Loading data...')

df_spy = pd.read_json('https://api.iextrading.com/1.0/stock/'
    +'spy'+'/chart/5y')
df_ind = pd.read_json('https://api.iextrading.com/1.0/stock/'
    +'qtec'+'/chart/5y')
df_stock = pd.read_json('https://api.iextrading.com/1.0/stock/'
    +'amzn'+'/chart/5y')

#how far back to calculate covariance
days_back = 20
print('==>Calculating CGRs...')
def calc_cgrs():
    spy_price = np.array(df_spy.close)
    spy_pricep1 = np.roll(spy_price,1)
    spy_ratio = spy_pricep1/spy_price
    spy_cgr = np.log(spy_ratio)[1:]

    ind_price = np.array(df_ind.close)
    ind_pricep1 = np.roll(ind_price,1)
    ind_ratio = ind_pricep1/ind_price
    ind_cgr = np.log(ind_ratio)[1:]

    stock_price = np.array(df_stock.close)
    stock_pricep1 = np.roll(stock_price,1)
    stock_ratio = stock_pricep1/stock_price
    stock_cgr = np.log(stock_ratio)[1:]

    stock_volume = np.array(df_stock.volume)[1:]

    
    return stock_cgr, stock_volume, spy_cgr, ind_cgr

stock_cgr, stock_volume, spy_cgr, ind_cgr = calc_cgrs()
#ignore this one #first row of X is first nke_cgr, first nke_volume, first spy_cgr
#first row of X is all nke_cgr, seoncd is volume, thrid is spy
X_full = np.vstack((stock_cgr, stock_volume, spy_cgr, ind_cgr))

print('==>Making Bins...')
#make 5 bins: big loss, small loss, very small loss/gain, small gain, big gain
returns = sorted(stock_cgr)
gains = [i for i in returns if i>float(0)]
losses = [i for i in returns if i<=float(0)]
num_gains = len(gains)
num_losses = len(losses)

bins = []
big_loss_bin = (-float("inf"), losses[num_losses//3])
small_loss_bin = (losses[(num_losses//3) + 1], losses[(num_losses*5)//6])
mid_bin = (losses[1+(num_losses*5)//6], gains[num_gains//6])
small_gain_bin = (gains[1+num_gains//6], gains[(num_gains*2)//3])
big_gain_bin = (gains[1+(num_gains*2)//3], float("inf"))

bins = [big_loss_bin, small_loss_bin, mid_bin, small_gain_bin, big_gain_bin]

def make_label(ret, labels="bins"):
    if labels=='bins':
        for i in range(5):
            (low, high) = bins[i]
            if low <= ret and ret <= high:
                return i
        #raise ValueError("unreal cgr, could not label")
    elif labels=='binary':
        if ret < 0:
            return 0
        else:
            return 1

#make indices for sampling
#inds = np.arange(0, len(nke_cgr), days_back)[:-1]
X_len = len(X_full[0])-days_back-1
X_samples = np.zeros((X_len, 4, days_back))
X_covs = np.zeros((X_len, 4, 4))
X_evals = np.zeros((X_len, 4))
X_evecs_ = np.zeros((X_len, 4, 4))

labels = np.zeros(X_len)

#normalize volume
minvol = np.min(X_full[:,1])
maxvol = np.max(X_full[:,1])
X_full[:,1] -= minvol
X_full[:,1] /= (maxvol-minvol)

print('==>Generating Samples and Labels...')

for i in range(X_len):
    X_samples[i] = X_full[:,i:(i+days_back)].reshape((4, days_back))
    
    #compute covariance
    X_covs[i] = np.cov(X_samples[i])
    #compute eigenvalues
    X_evals[i], X_evecs_[i] = np.linalg.eigh(X_covs[i])
    X_evecs = X_evecs_.reshape((X_len, 16))
    labels[i] = make_label(X_full[0,i+days_back], labels='binary')

#print(X_covs)
print("Generated "+str(X_len)+" samples")