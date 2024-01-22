"""
Python implementation of ML for a very simple LMM with a single random effect and a single intercept fixed effect.
"""

from numbers import Integral, Real
from typing import Union
import numpy

def neg2LogLik(
        varComp: Union[numpy.ndarray,tuple,list],
        y: numpy.ndarray,
        V: numpy.ndarray,
        d: numpy.ndarray,
        n: Integral = None,
    ) -> Real:
    """
    -2 log-likelihood function for ML estimation
    
    Parameters
    ----------
    varComp : numpy.ndarray, tuple, list
        A vector of shape (2,) containing parameter estimates (varE,varU).
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
    # (nobs,) - scalar -> (nobs,)
    y = y - y.mean()
    # calculate V.T @ y to get eta values
    # (ncomp,nobs) @ (nobs,) -> (ncomp,)
    Vy = V.T @ y
    # square eta values
    # (ncomp,)^2 -> (ncomp,)
    Vy2 = Vy**2
    # extract variance components
    # (2,)[i] -> scalar
    varE = varComp[0]
    varU = varComp[1]
    # calculate the ratio between genetic variance and error variance
    lamb = varU / varE
    # calculate diagonal values for varU * G + varE * I matrix spectral decomposition
    dStar = (d*lamb + 1)
    # calculate log-determinant using sum of logs of diagonal matrix
    sumLogD = numpy.sum(numpy.log(dStar))
    # calculate log-likelihood in two parts
    neg2LogLik_1 = (n * numpy.log(varE) + sumLogD )
    # calculate log-likelihood in two parts
    neg2LogLik_2 = numpy.sum(Vy2 / dStar) / varE
    # calculate log-likelihood
    out = neg2LogLik_1 + neg2LogLik_2
    # return log-likelihood
    return out

def calc_etasq(
        y: numpy.ndarray,
        V: numpy.ndarray,
    ) -> numpy.ndarray:
    """
    Calculate the eta squared vector from y and V
    
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
    # (nobs,) - scalar -> (nobs,)
    y = y - y.mean()
    # calculate V.T @ y to get eta values
    # (ncomp,nobs) @ (nobs,) -> (ncomp,)
    eta = V.T @ y
    # square eta values
    # (ncomp,)^2 -> (ncomp,)
    etasq = eta**2
    # return values
    return etasq

def neg2LogLik_v2(
        varComp: Union[numpy.ndarray,tuple,list],
        etasq: numpy.ndarray,
        d: numpy.ndarray,
        n: Integral = None,
    ) -> Real:
    """
    -2 log-likelihood function for ML estimation
    
    Parameters
    ----------
    varComp : numpy.ndarray, tuple, list
        A vector of shape (2,) containing parameter estimates (varE,varU).
    etasq : numpy.ndarray
        A vector of shape (ncomp,) containing eta squared values.
    d : numpy.ndarray
        A vector of shape (ncomp,) containing non-zero eigenvalues for the genomic relationship matrix.
    n : Integral
        Number of observations (nobs).
    
    Returns
    -------
    out : Real
        Scalar value proportional to the -2 log-likelihood for ML estimation.
    """
    # extract variance components
    # (2,)[i] -> scalar
    varE = varComp[1]
    varU = varComp[2]
    # calculate the ratio between genetic variance and error variance
    lamb = varU / varE
    # calculate diagonal values for varU * G + varE * I matrix spectral decomposition
    dStar = (d * lamb + 1)
    # calculate log-determinant using sum of logs of diagonal matrix
    sumLogD = sum(numpy.log(dStar))
    # calculate log-likelihood in two parts
    neg2LogLik_1 = (n * numpy.log(varE) + sumLogD )
    # calculate log-likelihood in two parts
    neg2LogLik_2 = numpy.sum(etasq / dStar) / varE
    # calculate log-likelihood
    out = neg2LogLik_1 + neg2LogLik_2
    # return log-likelihood
    return out

