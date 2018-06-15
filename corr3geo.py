#corr3geo.py

#We plot and compare geodesic curves in Corr(3) using different correction
#rules to ensure elements stay within the manifold.
#6/6/18
import numpy as np
import scipy as sc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plot3dEllipsoids.py as p3e

def corrProj(spd):
	D = np.diag(np.diag(SPD))
	Dhalf = sc.linalg.sqrtm(D);
	Result = np.divide(np.divide(SPD, Dhalf), Dhalf)
	return Result


#initial conditions and update rule
start = np.eye(3)
A = 4*np.random.uniform(size=(3,3))-2
Tan = 0.5*(A + A.T)
for i in np.arange(0,8,4):
	Tan[i]=0
T = np.arange(0,1,0.02)
L = len(T)


def update(s):
	sqstart = sc.linalg.sqrtm(start)
	mid = sc.linalg.expm(s*(np.divide(np.divide(Tan, sqstart), sqstart)))
	return sqstart*mid*sqstart

GeoNorm = np.zeros((3,3,L)) # This set of matrices will normalize at each step.
GeoOnes = np.zeros((3,3,L)) # This set of matrices will simply set diagonals to one at each step.
GeoDrift= np.zeros((3,3,L)) # This set of matrices will leave the terms unchanged after update.

PointsNorm = np.zeros((L,3)) # This will contain off-diags for GeoNorm
PointsOnes = np.zeros((L,3)) # This will contain off-diags for GeoOnes
PointsDrift= np.zeros((L,3)) # This will contain off-diags for GeoDrift

## Compute Geodesic Elements
for i in range(L):
    # Update GeoNorm
    New     = update(T[i])
    GeoDrift[:,:,i] = New
    PointsDrift[i,:]= [GeoDrift[0,1,i], GeoDrift[0,2,i], GeoDrift[1,2,i]
    
    D       = np.diag(np.diag(New));
    GeoNorm(:,:,i) = corrProj(New);
    PointsNorm[i,:]= [GeoNorm[0,1,i], GeoNorm[0,2,i], GeoNorm[1,2,i]
    
    # Update GeoOnes
    NewOnes = New 
    for i in np.arange(0,8,4):
		NewOnes[i] = 1
    GeoOnes[:,:,i] = NewOnes
    
    PointsOnes[i,:] = [GeoOnes[0,1,i], GeoOnes[0,2,i], GeoOnes[1,2,i]
    


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(PointsNorm[:,0], PointsNorm[:,1], PointsNorm[:,2], label='or')
ax.scatter(PointsDrift[:,0], PointsDrift[:,1], PointsDrift[:,2], label='ob')
# For each set of style and range settings, plot n random points in the box


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('Geodesic Curves in Corr(3)')
plt.legend()
#plt.show()

subarr = np.arange(0, L, 5)

p3e.plot3DEllipsoids(GeoNorm[:,:,subarr], 'COR', 'Correlation Geodesic (Normed Method)')
#plot3DEllipsoids(GeoOnes, 'COR', 'Correlation Geodesic (Reset Diagonal)')
p3e.plot3DEllipsoids(GeoDrift[:,:,subarr], 'COR', 'Correlation Geodesic (Numeric Drift)')
   
plt.show()




















