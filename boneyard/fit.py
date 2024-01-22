"""
Module containing fitting and utility functions for RR-BLUP.
"""

__all__ = [
    "check_is_ndarray",
    "check_ndarray_ndim",
    "check_ndarray_axis_len_eq",
    "rrBLUP_REML_calc_S",
    "rrBLUP_REML_calc_SZZtS",
    "rrBLUP_REML_calc_lamb_Ur",
    "rrBLUP_REML_calc_nusq",
    "rrBLUP_REML_logLik",
    "rrBLUP_REML_logLik_deriv",
    "rrBLUP_REML_grid_search",
    "rrBLUP_REML",
]

import math
from numbers import Integral, Real
from typing import Tuple
import numpy

from rrblup.util import check_is_ndarray, check_ndarray_ndim, check_ndarray_axis_len_eq

def rrBLUP_REML_calc_S(X: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate the null space projection matrix for REML calculations. The null 
    space projection matrix projects data such that fixed effects are 
    transformed to having zero effect as required by REML methodology. This 
    allows for more accurate estimation of variance components.

    This is a bare-bones function and assumes input types are correct.

    The null space projection matrix is computed using QR decomposition.
    
    Parameters
    ----------
    X : numpy.ndarray
        A matrix of shape ``(n,p)`` containing fixed effect predictors.

        Where:

        - ``n`` is the number of observations.
        - ``p`` is the number of fixed effect predictors.
    
    Returns
    -------
    out : numpy.ndarray
        A matrix of shape ``(n,n)`` designating the null space projection.
        The null space projection matrix is symmetric and idempotent.

        Where:

        - ``n`` is the number of observations.
    """
    # make sure fixed effects are not overdetermined
    if X.shape[0] < X.shape[1]:
        raise ValueError("'X' is overdetermined")
    # compute QR decomposition of X
    # (nobs,nfixed) -> (nobs,nobs), (nobs,nfixed) = Q, R
    Q, R = numpy.linalg.qr(X)
    # get the diagonal of the R matrix
    # (nobs,nfixed) -> (nfixed,)
    R_diag = numpy.diag(R)
    # if any of the diagonals are close to zero, raise error
    if numpy.any(numpy.absolute(R_diag) < 1e-12):
        raise ValueError("'X' is not or close to not full rank")
    # calculate the null space projection matrix for REML calculations
    # using the Q matrix to avoid matrix inversion
    # S = I - X (X'X)^-1 X'
    #   = I - QQ'
    # (nobs,nobs) - (nobs,nobs) @ (nobs,nobs) -> (nobs,nobs)
    S = numpy.identity(X.shape[0]) - (Q @ Q.T)
    return S

def rrBLUP_REML_calc_SZZtS(S: numpy.ndarray, Z: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate SZZ'S which is the model covariance structure.

    Parameters
    ----------
    S : numpy.ndarray
        A matrix of shape ``(n,n)`` designating the null space projection.
        The null space projection matrix is symmetric and idempotent.

        Where:

        - ``n`` is the number of observations.

    Z : numpy.ndarray
        A matrix of shape ``(n,q)`` containing random, genotypic marker effect predictors.
        This must be a genotypic matrix.
        
        Where:

        - ``n`` is the number of observations.
        - ``q`` is the number of genotypic markers.
    
    Returns
    -------
    out : numpy.ndarray
        A matrix of shape ``(n,n)`` representing the model covariance structure.

        Where:

        - ``n`` is the number of observations.
    """
    # calculate SZ
    # (n,n) @ (n,q) -> (n,q)
    SZ = S @ Z
    # calculate SZZ'S
    # since S = S' we can reduce the matrix muliplications using:
    # SZZ'S = SZZ'S' = (SZ)(SZ)'
    # (n,q) @ (q,n) = (n,n)
    SZZtS = SZ @ SZ.T
    return SZZtS

def rrBLUP_REML_calc_lamb_Ur(SHS: numpy.ndarray, nfixed: Integral) -> Tuple[numpy.ndarray,numpy.ndarray]:
    """
    Calculate the spectral decomposition of SHS and extract non-zero eigenvalues and eigenvectors.

    Parameters
    ----------
    SHS : numpy.ndarray
        A matrix of shape ``(n,n)`` representing the model covariance structure.

        Where:

        - ``n`` is the number of observations.

    nfixed : Integral
        The number of fixed effects for the REML model.
    
    Returns
    -------
    out : tuple[numpy.ndarray,numpy.ndarray]
        A tuple containing two elements: ``(lamb, Ur)``

        Where:

        - ``lamb`` is a vector of shape ``(nobs-nfixed,)`` containing non-zero eigenvalues.
        - ``Ur`` is a matrix of shape ``(nobs,nobs-nfixed)`` containing non-zero eigenvectors.
    """
    # Calculate the spectral decomposition for a symmetric matrix
    # SHS = U @ numpy.diag(D) @ U.T
    # We take the spectral decomposition since we know that we can add 
    # eigenvalues later:
    # ([Wr,Ur])diag(0, ..., 0, lambda_1 + delta, ..., lambda_n + delta)([Wr,Ur]')
    # (nobs,nobs) -> (n,), (nobs,nobs)
    D, U = numpy.linalg.eigh(SHS)
    # Get the non-zero eigenvalues
    # ``eigh`` stores eigenvalues increasing in order
    # (nobs,) -> (nobs-nfixed,)
    lamb = D[nfixed:]
    # Get the non-zero eigenvalue, eigenvectors by subsetting columns
    # (nobs,nobs) -> (nobs,nobs-nfixed)
    Ur = U[:,nfixed:]
    return lamb, Ur

def rrBLUP_REML_calc_nusq(Ur: numpy.ndarray, S: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate the nu matrix.
    Nu = Ur'SY
    NuSq = (Nu (Hadamard product) Nu)

    Parameters
    ----------
    Ur : numpy.ndarray
        A matrix of shape ``(nobs-nfixed,nobs)`` containing non-zero eigenvectors.

    S : numpy.ndarray
        A matrix of shape ``(nobs,nobs)`` designating the null space projection.
        The null space projection matrix is symmetric and idempotent.

    Y : numpy.ndarray
        A matrix of shape ``(nobs,ntrait)`` containing observations.
    """
    # Calculate nu squared values for the phenotypes
    # (nobs-nfixed,nobs) @ (nobs,nobs) @ (nobs,ntrait) -> (nobs-nfixed,ntrait)
    NuSq = numpy.square(Ur.T @ S @ Y)
    return NuSq

# Define restricted likelihood function to be optimized
def rrBLUP_REML_logLik(
        delta: Real, 
        nobs: Integral, 
        nfixed: Integral, 
        lamb: numpy.ndarray, 
        nusq: numpy.ndarray
    ) -> Real:
    """
    log-Likelihood function for RR-BLUP genomic prediction model using REML.
    This is a bare-bones function for maximum speed.

    Parameters
    ----------
    delta : Real
        The ratio between the environmental variance and genetic variance:
        delta = sigma_e^2 / sigma_g^2
        The variable for which we are trying to optimize.
        Value must be >= 0.0 since variance cannot be negative.
    nobs : Integral
        Number of observations.
    nfixed : Integral
        Number of fixed effects.
    lamb : numpy.ndarray
        An array of shape (nobs-nfixed,) containing eigenvalues.
    nusq : numpy.ndarray
        An array of shape (nobs-nfixed,) containing squared nu values.

    Returns
    -------
    out : Real
        The log-likelihood value for the given delta value.
    """
    df = nobs - nfixed  # calculate degrees of freedom
    lpd = lamb + delta  # calculate lambda + delta
    out = 0.5 * (
        df * numpy.log(df/(2.0*numpy.pi*numpy.e)) -
        df * numpy.log(numpy.sum(nusq/lpd)) -
        numpy.sum(numpy.log(lpd))
    )
    return out

# Define derivative of restricted likelihood function to be optimized
def rrBLUP_REML_logLik_deriv(
        delta: Real, 
        nobs: Integral, 
        nfixed: Integral, 
        lamb: numpy.ndarray, 
        nusq: numpy.ndarray
    ) -> Real:
    """
    Derivative of the log-Likelihood function for RR-BLUP genomic prediction model using REML.
    This is a bare-bones function for maximum speed.

    Parameters
    ----------
    delta : Real
        Variable for which we are trying to optimize.
    nobs : Integral
        Number of observations.
    nfixed : Integral
        Number of fixed effects.
    lamb : numpy.ndarray
        An array of shape (nobs-nfixed,) containing eigenvalues.
    nusq : numpy.ndarray
        An array of shape (nobs-nfixed,) containing squared nu values.

    Returns
    -------
    out : Real
        The derivative of the log-likelihood value for the given delta value.
    """
    df = nobs - nfixed  # calculate degrees of freedom
    lpd = lamb + delta  # calculate lambda + delta
    out = 0.5 * (
        df * numpy.sum(nusq/(lpd**2)) / numpy.sum(nusq/lpd) -
        numpy.sum(1.0/lpd)
    )
    return out

# Define Newton-Raphson search helper method
def rrblup_REML_NR_search(
        nobs: Integral, 
        nfixed: Integral, 
        lamb: numpy.ndarray, 
        nusq: numpy.ndarray,
        delta0: Real = 5e4,
        lbound: Real = 1e-8,
        ubound: Real = math.inf,
        tolerance: Real = 1e-8,
        maxiter: Real = 1e5,
    ) -> Tuple[Real,Real,Real]:
    """
    Perform a bounded Newton-Raphson search for REML RR-BLUP genomic prediction
    model parameter delta. Method uses the EMMA model specification.
    This is a bare-bonse function for maximum speed.

    Parameters
    ----------
    nobs : Integral
        Number of observations.
    nfixed : Integral
        Number of fixed effects.
    lamb : numpy.ndarray
        An array of shape (nobs-nfixed,) containing eigenvalues.
    nusq : numpy.ndarray
        An array of shape (nobs-nfixed,) containing squared nu values.
    delta0 : Real
        An initial guess of delta value to jump start the algorithm.
        Must be greater than or equal to zero.
    lbound : Real
        Lower search bound for delta values.
        Must be greater than or equal to zero.
    ubound : Real
        Upper search bound for delta values.
        Must be greater than or equal to zero.
        Can be infinity.
    tolerance : Real
        Tolerance threshold for relative change in delta values between each iteration.
        Must be greater than zero.
        Relative change is calculated as:

        relative change = abs( (delta_(i+1) - delta_(i)) / delta_(i) )

    maxiter : Real
        Maximum number of iterations to perform.
        
    Returns
    -------
    out : Tuple[Real,Real,Real]
        A tuple containing the optimal delta value, corresponding log-likelihood, 
        and derivative of the log likelihood value.
    """
    # calculate initial log-likelihood and its derivative for the initial delta value
    logLik0 = rrBLUP_REML_logLik(delta0, nobs, nfixed, lamb, nusq)
    logLik_deriv0 = rrBLUP_REML_logLik_deriv(delta0, nobs, nfixed, lamb, nusq)
    # calculate new updated guess (delta1)
    delta1 = delta0 - logLik0 / logLik_deriv0
    niter = 1
    # iterate until relative change tolerance achieved
    while abs((delta1 - delta0) / delta0) > tolerance and niter < maxiter:
        print("iter", niter, "::", "delta0=", delta0, "delta1=", delta1, "relchange=", abs((delta1 - delta0) / delta0))
        # update working guess (delta0) with new updated guess (delta1)
        delta0 = delta1
        # ensure that working guess (delta0) is within bounds
        delta0 = lbound if delta0 < lbound else delta0
        delta0 = ubound if delta0 > ubound else delta0
        # calculate log-likelihood and its derivative for working guess
        logLik0 = rrBLUP_REML_logLik(delta0, nobs, nfixed, lamb, nusq)
        logLik_deriv0 = rrBLUP_REML_logLik_deriv(delta0, nobs, nfixed, lamb, nusq)
        # calculate new updated guess (delta1)
        delta1 = delta0 - logLik0 / logLik_deriv0
        niter += 1
    # update final solution with new updated guess
    delta0 = delta1
    # calculate log-likelihood and its derivative for the final solution
    logLik0 = rrBLUP_REML_logLik(delta0, nobs, nfixed, lamb, nusq)
    logLik_deriv0 = rrBLUP_REML_logLik_deriv(delta0, nobs, nfixed, lamb, nusq)
    # return final solution, its log-likelihood, and its derivative
    return delta0, logLik0, logLik_deriv0

# Define grid search helper function
def rrBLUP_REML_grid_search(
        nobs: Integral, 
        nfixed: Integral, 
        lamb: numpy.ndarray, 
        nusq: numpy.ndarray,
        gridpts: Integral = 100,
        tolerance: Real = 1e-8,
    ) -> Tuple[Real,Real,Real]:
    """
    Perform a grid search for RR-BLUP genomic prediction model using REML and 
    the EMMA algorithm. This is a bare-bones function for maximum speed.

    Parameters
    ----------
    nobs : Integral
        Number of observations.
    nfixed : Integral
        Number of fixed effects.
    lamb : numpy.ndarray
        An array of shape (nobs-nfixed,) containing eigenvalues.
    nusq : numpy.ndarray
        An array of shape (nobs-nfixed,) containing squared nu values.

    Returns
    -------
    out : Tuple[Real,Real,Real]
        A tuple containing the optimal delta value, corresponding log-likelihood, 
        and derivative of the log likelihood value.
    """
    # generate grid in logarithmic space between
    # 1e-5 (near total environment) and 1e5 (near total genetic)
    grid = numpy.logspace(numpy.log(1e-5), numpy.log(1e5), gridpts, base=numpy.e)
    # for each point in the grid, calculate the derivative
    grid_deriv = numpy.empty(grid.shape, dtype=float)
    for i,value in enumerate(grid):
        grid_deriv[i] = rrBLUP_REML_logLik_deriv(value, nobs, nfixed, lamb, nusq)
    print(grid_deriv)
    # get lower and upper bounds for sliding window
    grid_lo = grid[0:gridpts-1]
    grid_up = grid[1:gridpts]
    # get the signs for each derivative
    grid_deriv_sign = numpy.sign(grid_deriv)
    print(grid_deriv_sign)
    # get the search interval indices
    search_ix = numpy.flatnonzero(grid_deriv_sign[0:gridpts-1] != grid_deriv_sign[1:gridpts])
    print(search_ix)
    # for each interval, calculate the optimal delta value
    delta = numpy.empty(len(search_ix), dtype=float)
    logLik = numpy.empty(len(search_ix), dtype=float)
    logLik_deriv = numpy.empty(len(search_ix), dtype=float)
    # for each interval, do a search using the Newton-Raphson method
    for ix in search_ix:
        # initialize as the midpoint between grid lower and upper bounds
        delta0 = 0.5 * (grid_lo[ix] + grid_up[ix])
        logLik0 = rrBLUP_REML_logLik(delta0, nobs, nfixed, lamb, nusq)
        logLik_deriv0 = rrBLUP_REML_logLik_deriv(delta0, nobs, nfixed, lamb, nusq)
        delta1 = delta0 - logLik0 / logLik_deriv0
        # iterate until tolerance achieved
        while abs(delta0 - delta1) < tolerance:
            delta0 = delta1 if delta1 > 0.0 else 0.0 # 
            logLik0 = rrBLUP_REML_logLik(delta0, nobs, nfixed, lamb, nusq)
            logLik_deriv0 = rrBLUP_REML_logLik_deriv(delta0, nobs, nfixed, lamb, nusq)
            delta1 = delta0 - logLik0 / logLik_deriv0
        # calculate likelihoods for delta the meets desired tolerance
        delta0 = delta1
        logLik0 = rrBLUP_REML_logLik(delta0, nobs, nfixed, lamb, nusq)
        logLik_deriv0 = rrBLUP_REML_logLik_deriv(delta0, nobs, nfixed, lamb, nusq)
        # save likelihood, derivative, delta values
        delta[ix] = delta0
        logLik[ix] = logLik0
        logLik_deriv[ix] = logLik_deriv0
    # identify the maximum log-likelihood index
    maxix = delta.argmax()
    # return values
    return delta[maxix], logLik[maxix], logLik_deriv[maxix]

def rrBLUP_REML(
        Y: numpy.ndarray,
        X: numpy.ndarray,
        Z: numpy.ndarray,
        gridpts: Integral = 100,
        tolerance: Real = 1e-8
    ):
    """
    Fit an RR-BLUP genomic prediction model using REML.
    
    Parameters
    ----------
    Y : numpy.ndarray
        A matrix of shape ``(n,t)`` containing observations.

        Where:

        - ``n`` is the number of observations.
        - ``t`` is the number of traits.
    
    X : numpy.ndarray
        A matrix of shape ``(n,p)`` containing fixed effect predictors.

        Where:

        - ``n`` is the number of observations.
        - ``p`` is the number of fixed effect predictors.

    Z : numpy.ndarray
        A matrix of shape ``(n,q)`` containing random, genotypic marker effect predictors.
        This must be a genotypic matrix.
        
        Where:

        - ``n`` is the number of observations.
        - ``q`` is the number of genotypic markers.
    
    Returns
    -------
    out : outtype
        outdesc
    """
    #############################
    ### Type and shape checks ###
    #############################

    # type checks
    check_is_ndarray(Y, "Y")
    check_is_ndarray(X, "X")
    check_is_ndarray(Z, "Z")

    # dimension checks
    check_ndarray_ndim(Y, "Y", 2)
    check_ndarray_ndim(X, "X", 2)
    check_ndarray_ndim(Z, "Z", 2)

    # get shape parameters from Y, X, Z
    nobs = Y.shape[0]   # n = nobs
    ntrait = Y.shape[1] # t = ntrait
    nfixed = X.shape[1] # p = nfixed
    nmkr = Z.shape[1]   # q = nmkr

    # check observation number
    check_ndarray_axis_len_eq(Y, "Y", 0, nobs)  # must be (nobs,ntrait)
    check_ndarray_axis_len_eq(X, "X", 0, nobs)  # must be (nobs,nfixed)
    check_ndarray_axis_len_eq(Z, "Z", 0, nobs)  # must be (nobs,nmkr)

    #############################
    ### Computational segment ###
    #############################

    # make sure fixed effects are not overdetermined
    if X.shape[0] < X.shape[1]:
        raise ValueError("'X' is overdetermined")

    # compute the null space projection matrix
    # (nobs,nfixed) -> (nobs,nobs)
    S = rrBLUP_REML_calc_S(X)

    # Corbeil and Searle (1976) proved that only a subset of rows in S is
    # required since many of these rows are linearly dependent.
    # If T is the row subset of S, then: SHS ~ THT'
    # In this scenario, we cannot subset rows of S as this gives incorrect
    # eigenvalues for SHS
    # T = S[0:(nobs-nfixed),:]
    # Calculate the REML model covariance
    SHS = rrBLUP_REML_calc_SZZtS(S, Z)

    # compute the non-zero eigenvalues and eigenvectors
    # SHS = S(ZKZ' + delta * I)S (K is identity in this case))
    # We only take SZKZ'S since we know that we can add in eigenvalues later:
    # ([Ur])diag(lambda_1 + delta, ..., lambda_n + delta)([Ur]')
    # (nobs,nobs), integer -> (nobs-nfixed,), (nobs,nobs-nfixed)
    lamb, Ur = rrBLUP_REML_calc_lamb_Ur(SHS, nfixed)

    # Calculate nu squared values for the phenotypes
    # (nobs,nobs-nfixed)' @ (nobs,nobs) @ (nobs,ntrait) -> (nobs-nfixed,ntrait)
    NuSq = rrBLUP_REML_calc_nusq(Ur, S, Y)

    # output vectors for optimized hyperparameters
    delta = numpy.empty(ntrait, dtype=float)
    logLik = numpy.empty(ntrait, dtype=float)
    logLik_deriv = numpy.empty(ntrait, dtype=float)

    # for each trait, perform a grid search
    for trait in range(ntrait):

        # subset nu**2 for trait
        nusq = NuSq[:,trait]

        # perform grid search for trait
        result = rrBLUP_REML_grid_search(nobs, nfixed, lamb, nusq, gridpts, tolerance)

        # store results
        delta[trait], logLik[trait], logLik_deriv[trait] = result

    return delta, logLik, logLik_deriv


