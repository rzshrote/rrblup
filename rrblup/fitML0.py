#!/usr/bin/env python3

from numbers import Integral, Real
from typing import Tuple, Union
import numpy
from scipy.optimize import minimize, Bounds

from rrblup.util import check_is_gteq
from rrblup.util import check_is_ndarray
from rrblup.util import check_is_Real
from rrblup.util import check_is_Integral
from rrblup.util import check_ndarray_ndim
from rrblup.util import check_ndarray_axis_len_eq

def rrBLUP_ML0_calc_G(Z: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate a genomic relationship matrix from a marker matrix.

    Parameters
    ----------
    Z : numpy.ndarray
        A genotype matrix of shape ``(nobs,nmkr)``.
    
    Returns
    -------
    G : numpy.ndarray
        A genomic relationship matrix of shape ``(nobs,nobs)``.
    """
    # copy and convert data to floating point representation
    # (nobs,nmkr) -> (nobs,nmkr)
    Z = Z.astype(float)
    # center each column around the mean marker value
    # (nobs,nmkr) - (1,nmkr) -> (nobs,nmkr)
    Z -= numpy.mean(Z, 0, keepdims = True)
    # scale each column by the standard deviation of the marker value
    # (nobs,nmkr) * (1,nmkr) -> (nobs,nmkr)
    Z *= (1.0 / numpy.std(Z, 0, keepdims = True))
    # take the outer product to get the genomic relationship matrix
    # (nobs,nmkr) @ (nmkr,nobs) -> (nobs,nobs)
    G = Z @ Z.T
    # scale the matrix by the mean diagonal value
    # (nobs,nobs) * scalar -> (nobs,nobs)
    G *= (1.0 / numpy.mean(numpy.diag(G)))
    # return genomic relationship matrix
    return G

def rrBLUP_ML0_center_y(y: numpy.ndarray) -> numpy.ndarray:
    """
    Center y values around zero.

    Parameters
    ----------
    y : numpy.ndarray
        A vector of observations of shape ``(nobs,)``.
    
    Returns
    -------
    out : numpy.ndarray
        A vector of observations of shape ``(nobs,)`` centered around zero.
    """
    # center trait data around mean
    # (nobs,) - scalar -> (nobs,)
    return y - y.mean()

def rrBLUP_ML0_calc_d_V(G: numpy.ndarray) -> Tuple[numpy.ndarray,numpy.ndarray]:
    """
    Calculate the spectral decomposition of a (symmetric) genomic relationship matrix.

    Parameters
    ----------
    G : numpy.ndarray
        A genomic relationship matrix of shape ``(nobs,nobs)``.
    
    Returns
    -------
    out : tuple
        A tuple containing ``(d,V)``.
        
        Where::
        - ``d`` is a vector of shape ``(nobs,)`` representing the diagonal of the spectral decomposition.
        - ``V`` is a matrix of shape ``(nobs,nobs)`` representing the orthonormal basis of the spectral decomposition.

        Entries are sorted from highest eigenvalue to lowest eigenvalue.
    """
    # take the spectral decomposition of the (symmetric) G matrix.
    # (nobs,nobs) -> eigenvalues = (nobs,), eigenvectors = (nobs,nobs)
    d, V = numpy.linalg.eigh(G)
    # get indices for highest to lowest eigenvalues
    # (nobs,)
    ix = numpy.argsort(d)[::-1]
    # reorder eigenvalues from highest to lowest
    # (nobs,)[(nobs,)] -> (nobs,)
    d = d[ix]
    # reorder eigenvectors from highest to lowest
    # (nobs,)[(nobs,)] -> (nobs,)
    V = V[:,ix]
    # return spectral decomposition
    return d, V

def rrBLUP_ML0_nonzero_d_V(d: numpy.ndarray, V: numpy.ndarray, tol: Real = 1e-5) -> Tuple[numpy.ndarray,numpy.ndarray]:
    """
    Extract nonzero components of eigenvalues and eigenvectors from a spectral decomposition.

    Parameters
    ----------
    d : numpy.ndarray
        A vector of shape ``(nobs,)`` representing the diagonal of the spectral decomposition.
    V : numpy.ndarray
        A matrix of shape ``(nobs,nobs)`` representing the orthonormal basis of the spectral decomposition.
    
    Returns
    -------
    out : tuple
        A tuple containing ``(d,V)``.
        
        Where::
        - ``d`` is a vector of shape ``(ncomp,)`` representing the diagonal of the spectral decomposition.
        - ``V`` is a matrix of shape ``(nobs,ncomp)`` representing the orthonormal basis of the spectral decomposition.
    """
    # for the spectral decomposition, construct a mask for eigenvalues greater than approximately zero
    # (nobs,) -> (nobs,) 
    mask = d > tol
    # extract eigenvalues that are greater than approximately zero
    # (nobs,) -> (ncomp,)
    d = d[mask]
    # extract eigenvectors for eigenvalues that are greater than approximately zero
    # (nobs,nobs) -> (nobs,ncomp)
    V = V[:,mask]
    # return values
    return d, V

def rrBLUP_ML0_calc_etasq(V: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate eta squared values for fast computation of likelihoods.

    Parameters
    ----------
    V : numpy.ndarray
        A matrix of shape ``(nobs,ncomp)`` containing the non-zero eigenvectors of the spectral decomposition.
    y : numpy.ndarray
        A vector of shape ``(nobs,)`` containing zero centered observations.
    
    Returns
    -------
    etasq : numpy.ndarray
        A vector of shape ``(ncomp,)`` containing squared eta values.
    """
    # calculate V.T @ y to get eta values
    # (ncomp,nobs) @ (nobs,) -> (ncomp,)
    eta = V.T @ y
    # square eta values
    # (ncomp,)^2 -> (ncomp,)
    etasq = eta**2
    # return values
    return etasq

def rrBLUP_ML0_neg2LogLik_fast(logVarComp: numpy.ndarray, etasq: numpy.ndarray, d: numpy.ndarray, n: Integral) -> Real:
    """
    -2 log-likelihood function for ML estimation.
    In optimization, this function is to be minimized.
    
    Parameters
    ----------
    logVarComp : numpy.ndarray
        A vector of shape (2,) containing parameter estimates in the log scale (log(varE),log(varU)).
        The log scale is used for more efficient search since the search space is (0,Inf).
    etasq : numpy.ndarray
        A vector of shape ``(ncomp,)`` containing squared eta values.
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
    # calculate -2 * log-likelihood
    n2ll = (n * numpy.log(varE) + sumLogD) + ((numpy.sum(etasq / dStar)) / varE)
    # return -2 * log-likelihood
    return n2ll

def rrBLUP_ML0_calc_ridge(varE: Real, varU: Real) -> Real:
    """
    Calculate the ridge parameter.

    Parameters
    ----------
    varE : Real
        Error variance.
    varU : Real
        Marker variance.
    
    Returns
    -------
    out : Real
        The ridge parameter.
    """
    return varE / varU

def rrBLUP_ML0_calc_ZtZplI(Z: numpy.ndarray, ridge: Real) -> numpy.ndarray:
    """
    Calculate (Z'Z + lambda * I).

    Parameters
    ----------
    Z : numpy.ndarray
        A genotype matrix of shape ``(nobs,nmkr)``.
    ridge : Real
        The ridge parameter, lambda. Must be non-negative.

    Returns
    -------
    out : numpy.ndarray
        The calculated matrix of shape ``(nmkr,nmkr)``.
    """
    # calculate marker covariance matrix: Z'Z
    # (nmkr,nobs) @ (nobs,nmkr) -> (nmkr,nmkr)
    ZtZplI = Z.T @ Z
    # extract a view of the diagonal for which to add the ridge parameter
    diagZtZplI = numpy.einsum('ii->i', ZtZplI)
    # add ridge parameter to diagonal of Z'Z
    # (nmkr,) + scalar -> (nmkr,)
    diagZtZplI += ridge
    # return values
    return ZtZplI

def rrBLUP_ML0_calc_Zty(Z: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate Z'y.
    
    Parameters
    ----------
    Z : numpy.ndarray
        A genotype matrix of shape ``(nobs,nmkr)``.
    y : numpy.ndarray
        A vector of shape ``(nobs,)`` containing zero centered observations.

    Returns
    -------
    out : numpy.ndarray
        A vector of shape ``(nmrk,)``.
    """
    # calculate Z'y
    # (nmkr,nobs) @ (nobs,) -> (nmkr,)
    Zty = Z.T @ y
    # return values
    return Zty

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

def rrBLUP_ML0(y: numpy.ndarray, Z: numpy.ndarray, varlb: Real = 1e-5, varub: Real = 1e5, gsatol: Real = 1e-8, gsmaxiter: Integral = 1000):
    """
    Ridge regression BLUP for the simple model::

    y = Zu + e

    Where::
        - ``y`` are observations.
        - ``Z`` is a design matrix for genetic markers.
        - ``u`` are marker effects which follow the distribution ``MVN(0, varU * I)``.
        - ``e`` are errors which follow the distribution ``MVN(0, varE * I)``.
    
    Uses the EMMA formulation to solve for ``varE`` and ``varU``.
    Uses the Nelder-Mead method to optimize for variance components.
    Marker effects are estimated using the Gauss-Seidel method.

    Parameters
    ----------
    y : numpy.ndarray
        A vector of observations of shape ``(nobs,)``. If not mean centered, will be centered around zero.
    Z : numpy.ndarray
        A genotype matrix of shape ``(nobs,nmkr)``.
    varlb : Real
        Lower bound permitted for variance component estimation.
        Must be non-negative.
    varub : Real
        Upper bound permitted for variance component estimation.
        Must be non-negative and greater than ``varlb``.
    gsatol : Real
        Absolute tolerance for the Gauss-Seidel method.
        Iterate until the sum of absolute differences between successive 
        iterations is less than this value or ``maxiter`` is reached.
        Must be non-negative.
    gsmaxiter : Integral
        Maximum number of iterations for the Gauss-Seidel method.
        Must be non-negative.

    Returns
    -------
    out : dict
        A dictionary of output values.
    """
    # check input types
    check_is_ndarray(y, "y")
    check_is_ndarray(Z, "Z")
    check_is_Real(varlb, "varlb")
    check_is_Real(varub, "varub")
    check_is_Real(gsatol, "gsatol")
    check_is_Integral(gsmaxiter, "gsmaxiter")

    # check input values
    check_ndarray_ndim(y, "y", 1)
    check_ndarray_ndim(Z, "Z", 2)
    check_ndarray_axis_len_eq(Z, "Z", 0, len(y))
    check_is_gteq(varlb, "varlb", 0.0)
    check_is_gteq(varub, "varub", varlb)
    check_is_gteq(gsatol, "gsatol", 0.0)
    check_is_gteq(gsmaxiter, "gsmaxiter", 0)

    # get the mean of y (the intercept)
    # (nobs,) -> scalar
    meanY = y.mean()

    # center trait data around mean
    # (nobs,)
    y = rrBLUP_ML0_center_y(y)

    # get the number of observations
    # scalar
    nobs = len(y)

    # create genomic relationship matrix
    # (nobs,nobs)
    G = rrBLUP_ML0_calc_G(Z)

    # take the spectral decomposition of the (symmetric) G matrix.
    # (nobs,nobs) -> eigenvalues = (nobs,), eigenvectors = (nobs,nobs)
    d, V = rrBLUP_ML0_calc_d_V(G)

    # remove zero components
    # eigenvalues = (nobs,), eigenvectors = (nobs,nobs) -> eigenvalues = (ncomp,), eigenvectors = (nobs,ncomp)
    d, V = rrBLUP_ML0_nonzero_d_V(d, V)

    # calculate eta squared values
    # (ncomp,)
    etasq = rrBLUP_ML0_calc_etasq(V, y)

    # calculate variance of y
    # (nobs,) -> scalar
    varY = y.var()

    # calculate initial estimates of log(varE) and log(varU); set each to half of varY
    # scalar
    logVarE0 = numpy.log(0.5 * varY)
    logVarU0 = numpy.log(0.5 * varY)

    # construct inital starting position
    # (2,)
    logVarComp0 = numpy.array([logVarE0, logVarU0])

    # construct search space boundaries
    bounds = Bounds(
        lb = numpy.repeat(numpy.log(varlb), len(logVarComp0)),
        ub = numpy.repeat(numpy.log(varub), len(logVarComp0)),
    )

    # optimize for the variance components using Nelder-Mead algorithm
    soln = minimize(
        fun = rrBLUP_ML0_neg2LogLik_fast,
        x0 = logVarComp0,
        args = (etasq, d, nobs),
        method = 'Nelder-Mead',
        bounds = bounds,
    )

    # get the solution values
    varE = numpy.exp(soln.x[0])
    varU = numpy.exp(soln.x[1])

    # calculate the ridge parameter
    ridge = rrBLUP_ML0_calc_ridge(varE, varU)

    # calculate (Z'Z + lambda * I)
    ZtZplI = rrBLUP_ML0_calc_ZtZplI(Z, ridge)

    # calculate Z'y
    Zty = rrBLUP_ML0_calc_Zty(Z, y)

    # solve for (Z'Z + lambda * I)u = Z'y using the Gauss-Seidel method
    uhat = gauss_seidel(ZtZplI, Zty, gsatol, gsmaxiter)

    # calculate heritability
    h2 = varU / (varU + varE)

    # reconstruct the log-likelihood (minus constant)
    logLik = -2 * soln.fun

    # get intercept incidence matrix
    # (nobs,1)
    X = numpy.full((nobs,1), 1.0, float)

    # get intercept
    # (1,)
    betahat = numpy.array([meanY])

    # calculate y hat
    yhat = X.dot(betahat) + Z.dot(uhat)

    # create output dictionary
    out = {
        # ridge regression elements
        "yhat": yhat,
        "X": X,
        "betahat": betahat,
        "Z": Z,
        "uhat": uhat,
        # variance estimates
        "varE": varE,
        "varU": varU,
        "h2": h2,
        "logLik": logLik,
        # optimization solution object
        "soln": soln,
    }

    return out
