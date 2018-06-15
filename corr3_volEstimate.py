#corr3_volEstimate.py
import math
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
# This script tries to empirically confirm the theoretical volume of the
# covariance manifold of dimension 3.  We do this using a random sampling
# method.
#6/6/18

N=10000
trials=1000
#cols = np.zeros((N,3))

def low(x,y):
	return x*y - math.sqrt((1-x**2)*(1-y**2))

def high(x,y):
	return x*y + math.sqrt((1-x**2)*(1-y**2))

estimates = np.zeros((trials,1))
for j in range(trials):
	count = 0
	for i in range(N):
		x = 2*np.random.uniform() - 1
		y = 2*np.random.uniform() - 1
		z = 2*np.random.uniform() - 1
		if z>low(x,y) and x<high(x,y):
			#cols[i,:]=np.array([1,0,0])
			count += 1
		#else:
			#cols[i,:]=np.array([0,0,0])
		#plot x,y,z,'marker','o','color', cols[i,:]

	estimates[j] = float(count)*8/N

#proportion of points in manifold time volume of cube
theory = 0*np.pi**2
avg = np.mean(estimates)
std = np.std(estimates)


