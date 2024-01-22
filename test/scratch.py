#!/usr/bin/env python3

from numbers import Integral, Real
from typing import Union
import numpy
import pandas
from matplotlib import pyplot

#####################
### Initial tests ###
#####################

# load data
traits_df = pandas.read_csv("traits.csv")
markers_df = pandas.read_csv("markers.csv")
# Gmatrix_df = pandas.read_csv("Gmatrix.csv")
# Geigenvectors_df = pandas.read_csv("Geigenvectors.csv")
# Geigenvalues_df = pandas.read_csv("Geigenvalues.csv")

# get data
Z = markers_df.to_numpy(dtype = float)
y = traits_df.iloc[:,1].to_numpy(dtype = float)

# create genomic relationship matrix
# (nobs,nobs)
Z -= numpy.mean(Z, 0, keepdims = True) # (nobs,nmkr)
Z *= (1.0 / numpy.std(Z, 0, keepdims = True))
G = Z @ Z.T
G *= (1.0 / numpy.mean(numpy.diag(G)))

# take the spectral decomposition of the (symmetric) G matrix.
# (nobs,nobs) -> eigenvalues = (nobs,), eigenvectors = (nobs,nobs)
d, V = numpy.linalg.eigh(G)
ix = numpy.argsort(d)[::-1] # sort highest to lowest eigenvalue
d = d[ix]
V = V[:,ix]

# center trait data around mean
# (nobs,)
y = y - y.mean()

# for the spectral decomposition, construct a mask for eigenvalues greater than approximately zero
# (nobs,) -> (nobs,) 
mask = d > 1e-5

# extract eigenvalues that are greater than approximately zero
# (nobs,) -> (ncomp,)
d = d[mask]

# extract eigenvectors for eigenvalues that are greater than approximately zero
# (nobs,nobs) -> (nobs,ncomp)
V = V[:,mask]

# calculate V.T @ y to get eta values
# (ncomp,nobs) @ (nobs,) -> (ncomp,)
eta = V.T @ y

# square eta values
# (ncomp,)^2 -> (ncomp,)
etasq = eta**2

# variance components
varE = 0.5725375
varU = 0.4690475

# calculate the ratio between genetic variance and error variance
lamb = varU / varE

# calculate diagonal values for varU * G + varE * I matrix spectral decomposition
dStar = (d*lamb + 1)

# calculate log-determinant using sum of logs of diagonal matrix
sumLogD = numpy.sum(numpy.log(dStar))

# calculate log-likelihood in two parts
neg2LogLik_1 = (len(y) * numpy.log(varE) + sumLogD )

# calculate log-likelihood in two parts
neg2LogLik_2 = (numpy.sum(etasq / dStar)) / varE

# calculate log-likelihood
n2ll = neg2LogLik_1 + neg2LogLik_2

h2 = varU / (varU + varE)

ll = -2.0 * n2ll

print(varE, varU, h2, ll)

############################################################
### Development of a slow likelihood evaluation function ###
############################################################

def neg2LogLik(
        logVarComp: Union[numpy.ndarray,tuple,list],
        y: numpy.ndarray,
        V: numpy.ndarray,
        d: numpy.ndarray,
        n: Integral = None,
    ) -> Real:
    """
    -2 log-likelihood function for ML estimation
    
    Parameters
    ----------
    logVarComp : numpy.ndarray, tuple, list
        A vector of shape (2,) containing parameter estimates in the log scale (log(varE),log(varU)).
        The log scale is used for more efficient search since the search space is (0,Inf).
    y : numpy.ndarray
        A vector of shape (nobs,) containing observations.
    V : numpy.ndarray
        A matrix of shape (nobs,ncomp) containing non-zero eigenvectors for the genomic relationship matrix.
    d : numpy.ndarray
        A vector of shape (ncomp,) containing non-zero eigenvalues for the genomic relationship matrix.
    n : Integral
        Number of observations (nobs).
    
    Returns
    -------
    out : Real
        Scalar value proportional to the -2 log-likelihood for ML estimation.
    """
    # center y values around their mean; this removes the effect of the intercept
    # (nobs,)
    y = y - y.mean()
    # calculate V.T @ y to get eta values
    # (ncomp,nobs) @ (nobs,) -> (ncomp,)
    eta = V.T @ y
    # square eta values
    # (ncomp,)^2 -> (ncomp,)
    etasq = eta**2
    # variance components
    varE = numpy.exp(logVarComp[0])
    varU = numpy.exp(logVarComp[1])
    # calculate the ratio between genetic variance and error variance
    lamb = varU / varE
    # calculate diagonal values for varU * G + varE * I matrix spectral decomposition
    dStar = (d * lamb + 1)
    # calculate log-determinant using sum of logs of diagonal matrix
    sumLogD = numpy.sum(numpy.log(dStar))
    # calculate log-likelihood in two parts
    neg2LogLik_1 = (n * numpy.log(varE) + sumLogD)
    # calculate log-likelihood in two parts
    neg2LogLik_2 = (numpy.sum(etasq / dStar)) / varE
    # calculate log-likelihood
    n2ll = neg2LogLik_1 + neg2LogLik_2
    return n2ll

_n2ll = neg2LogLik((numpy.log(varE),numpy.log(varU)), y, V, d, len(y))
_ll = -2.0 * _n2ll

############################################################
### Development of a fast likelihood evaluation function ###
############################################################

def calc_etasq(
        y: numpy.ndarray,
        V: numpy.ndarray,
    ) -> numpy.ndarray:
    """
    Calculate eta squared values for faster log-likelihood evaluation.
    
    Parameters
    ----------
    y : numpy.ndarray
        A vector of shape (nobs,) containing observations.
    V : numpy.ndarray
        A matrix of shape (nobs,ncomp) containing non-zero eigenvectors for the genomic relationship matrix.
    
    Returns
    -------
    etasq : numpy.ndarray
        A vector of shape (ncomp,) containing eta squared values.
    """
    # center y values around their mean; this removes the effect of the intercept
    # (nobs,)
    y = y - y.mean()
    # calculate V.T @ y to get eta values
    # (ncomp,nobs) @ (nobs,) -> (ncomp,)
    eta = V.T @ y
    # square eta values
    # (ncomp,)^2 -> (ncomp,)
    etasq = eta**2
    return etasq

def neg2LogLik_fast(
        logVarComp: Union[numpy.ndarray,tuple,list],
        etasq: numpy.ndarray,
        d: numpy.ndarray,
        n: Integral = None,
    ) -> Real:
    """
    -2 log-likelihood function for ML estimation
    
    Parameters
    ----------
    logVarComp : numpy.ndarray, tuple, list
        A vector of shape (2,) containing parameter estimates in the log scale (log(varE),log(varU)).
        The log scale is used for more efficient search since the search space is (0,Inf).
    y : numpy.ndarray
        A vector of shape (nobs,) containing observations.
    V : numpy.ndarray
        A matrix of shape (nobs,ncomp) containing non-zero eigenvectors for the genomic relationship matrix.
    d : numpy.ndarray
        A vector of shape (ncomp,) containing non-zero eigenvalues for the genomic relationship matrix.
    n : Integral
        Number of observations (nobs).
    
    Returns
    -------
    out : Real
        Scalar value proportional to the -2 log-likelihood for ML estimation.
    """
    # variance components
    varE = numpy.exp(logVarComp[0])
    varU = numpy.exp(logVarComp[1])
    # calculate the ratio between genetic variance and error variance
    lamb = varU / varE
    # calculate diagonal values for varU * G + varE * I matrix spectral decomposition
    dStar = (d * lamb + 1)
    # calculate log-determinant using sum of logs of diagonal matrix
    sumLogD = numpy.sum(numpy.log(dStar))
    # calculate log-likelihood in two parts
    neg2LogLik_1 = (n * numpy.log(varE) + sumLogD)
    # calculate log-likelihood in two parts
    neg2LogLik_2 = (numpy.sum(etasq / dStar)) / varE
    # calculate log-likelihood
    n2ll = neg2LogLik_1 + neg2LogLik_2
    return n2ll

_etasq = calc_etasq(y, V)
_n2ll = neg2LogLik_fast((numpy.log(varE),numpy.log(varU)), _etasq, d, len(y))
_ll = -2.0 * _n2ll

############################################
### Plotting of -2 log-likelihood values ###
############################################

pts = numpy.linspace(-1, 0, 30)
gridpts = numpy.meshgrid(pts, pts)

etasq = calc_etasq(y, V)

gridX = gridpts[0] # (g,g) containing log(varE) values
gridY = gridpts[1] # (g,g) containing log(varU) values
gridZ = numpy.empty(gridX.shape, dtype = float)
for i in range(gridX.shape[0]):
    for j in range(gridX.shape[1]):
        gridZ[i,j] = neg2LogLik_fast((gridX[i,j],gridY[i,j]), etasq, d, len(y))

fig, ax = pyplot.subplots()
CS = ax.contour(gridX, gridY, gridZ, levels = 10)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_xlabel("log(varE)")
ax.set_ylabel("log(varU)")
ax.set_title('-2 * log-likelihood (minimizing)')
pyplot.savefig("neg2LogLik.png")
pyplot.close()

##############################################################
### Calculate marker effects using the Gauss-Seidel method ###
##############################################################

# calculate marker covariance matrix: Z'Z
# (nmkr,nobs) @ (nobs,nmkr) -> (nmkr,nmkr)
ZtZ = Z.T @ Z

# extract a view of the diagonal for which to add (varE / varU)
# ZtZ.ravel()[::ZtZ.shape[1]+1] += (varE / varU)
diagZtZ = numpy.einsum('ii->i', ZtZ)

# add ridge parameter to diagonal of Z'Z
# (nmkr,) + scalar -> (nmkr,)
diagZtZ += (varE / varU)

# calculate Z'y
# (nmkr,nobs) @ (nobs,) -> (nmkr,)
Zty = Z.T @ y

def gauss_seidel(A: numpy.ndarray, b: numpy.ndarray, atol: Real = 1e-8, maxiter: Integral = 1000) -> numpy.ndarray:
    """
    Solve the equation Ax = b using the Gauss-Seidel method.

    Parameters
    ----------
    A : numpy.ndarray
        A diagonal dominant or symmetric positive definite matrix of shape (nmkr,nmkr).
    b : numpy.ndarray
        A vector of shape (nmkr,).
    atol : Real
        Absolute tolerance. Iterate until the sum of absolute differences 
        between successive iterations is less than this value or ``maxiter``
        is reached.
        Must be non-negative.
    maxiter : Integral
        Maximum number of iterations.
    
    Returns
    -------
    x : numpy.ndarray
        Solution to the system of equations.
    """
    # get number of markers
    nmkr = len(b)
    # allocate memory for the previous x estimate
    xprev = numpy.zeros(nmkr, dtype = float)
    # allocate memory for the current x estimate
    xcurr = numpy.zeros(nmkr, dtype = float)
    # number of iterations
    niter = 0
    # absolute difference
    adiff = 2*atol
    # main loop
    while adiff > atol and niter < maxiter:
        # copy current x values to previous x values without memory allocation
        xprev[:] = xcurr
        # modify current x values using the Gauss-Seidel procedure
        for i in range(nmkr):
            xcurr[i] = (b[i] - A[i,:i].dot(xcurr[:i]) - A[i,i+1:].dot(xcurr[i+1:])) / A[i,i]
        # calculate the absolute difference
        adiff = numpy.sum(numpy.abs(xcurr - xprev))
        # increment the iteration number
        niter += 1
    # return estimates
    return xcurr

# get marker effect estimates
uhat = gauss_seidel(ZtZ, Zty)

# calculate errors
# (nobs,nmkr) @ (nmkr,) -> (nobs,)
yhat = Z @ uhat

# difference
# (nobs,) - (nobs,) -> (nobs,)
resid = y - yhat

# calculate r2 for training data
SST = numpy.sum(y**2)
SSE = numpy.sum(resid**2)
SSR = SST - SSE
r2 = SSR / SST

