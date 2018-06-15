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
df_nke = pd.read_json('https://api.iextrading.com/1.0/stock/'
    +'NKE'+'/chart/5y')

#how far back to calculate covariance
days_back = 20
print('==>Calculating CGRs...')
def calc_cgrs():
    spy_price = np.array(df_spy.close)
    spy_pricep1 = np.roll(spy_price,1)
    spy_ratio = spy_pricep1/spy_price
    spy_cgr = np.log(spy_ratio)[1:]

    nke_price = np.array(df_nke.close)
    nke_pricep1 = np.roll(nke_price,1)
    nke_ratio = nke_pricep1/nke_price
    nke_cgr = np.log(nke_ratio)[1:]

    nke_volume = np.array(df_nke.volume)[1:]

    
    return nke_cgr, nke_volume, spy_cgr

nke_cgr, nke_volume, spy_cgr = calc_cgrs()
#ignore this one #first row of X is first nke_cgr, first nke_volume, first spy_cgr
#first row of X is all nke_cgr, seoncd is volume, thrid is spy
X_full = np.vstack((nke_cgr, nke_volume, spy_cgr))

print('==>Making Bins...')
#make 5 bins: big loss, small loss, very small loss/gain, small gain, big gain
returns = sorted(nke_cgr)
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

def make_label(ret):
    for i in range(5):
        (low, high) = bins[i]
        if low <= ret and ret <= high:
            return i
    #raise ValueError("unreal cgr, could not label")

#make indices for sampling
#inds = np.arange(0, len(nke_cgr), days_back)[:-1]
X_len = len(X_full[0])-days_back-1
X_samples = np.zeros((X_len, 3, days_back))
X_covs = np.zeros((X_len, 3, 3))
X_eigs = np.zeros((X_len, 3))
labels = np.zeros(X_len)

print('==>Generating Samples and Labels...')

for i in range(X_len):
    X_samples[i] = X_full[:,i:(i+days_back)].reshape((3, days_back))
    #normalize volume
    minvol = np.min(X_samples[i,1])
    maxvol = np.max(X_samples[i,1])
    X_samples[i,1] -= minvol
    X_samples[i,1] /= maxvol-minvol
    #compute covariance
    X_covs[i] = np.cov(X_samples[i])
    #compute eigenvalues
    X_eigs[i] = np.linalg.eigvalsh(X_covs[i])
    labels[i] = make_label(X_full[0,i+days_back])

#print(X_covs)
print("Generated "+str(X_len)+" samples")