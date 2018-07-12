#SE3util.py
import numpy as np
import scipy as sc
import numpy.linalg as nlin
import scipy.linalg as slin

#uses metric inherited GA(n)
#based on paper: 
#https://repository.upenn.edu/cgi/viewcontent.cgi?article=1155&context=meam_papers
def MatVecToSEn(mat, vec):
    d1, d2 = mat.shape
    SEmat = np.zeros((d1+1, d2+1))
    SEmat[:d1,:d2] = mat
    SEmat[d1, :d2] = vec
    SEmat[d1+1, d2+1] = 1.
    return SE3mat

def SEnToMatVec(SEmat):
    d1, d2 = SEmat.shape
    mat = SEmat[:d1-1,:d2-1]
    vec = SEmat[d1-1, :d2-1]
    return mat, vec

def CovToMatVec(covmat):
    return nlin.eigh(covmat)

def CovToSEn(covmat):
    mat, vec = nlin.eigh(covmat)
    return MatVecToSEn(mat, vec)

def inner(Sx, Sy):
    return np.trace(Sx.T @ Sy)

def inner(wx, vx, wy, vy):
    return np.trace(wx.T @ wy) + np.dot(vx, vy)

def dist(B, A):
    diff = B - A
    return np.trace(diff.T @ diff)

#minimizes ||B-A||^2_GA wrt A

















