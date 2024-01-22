#!/usr/bin/env Rscript
# R implementation of ML and REML for a very simple LMM with a single random effect and a single intercept fixed effect.

# @brief    -2 log-likelihood function for ML estimation
# 
# @param    varComp
#           A vector of shape (2,) containing parameter estimates (varE,varU).
# @param    y
#           A vector of shape (nobs,) containing observations.
# @param    V
#           A matrix of shape (nobs,ncomp) containing non-zero eigenvectors for the genomic relationship matrix.
# @param    d
#           A vector of shape (ncomp,) containing non-zero eigenvalues for the genomic relationship matrix.
# @param    n
#           Number of observations (nobs).
#
# @return   out
#           Scalar value proportional to the -2 log-likelihood for ML estimation.
neg2LogLik_v2 <- function(
    varComp,
    y,
    V,
    d,
    n = length(y)
) {
    # center y values around their mean; this removes the effect of the intercept
    # (nobs,) - scalar -> (nobs,)
    y <- y - mean(y)
    # calculate V.T @ y to get eta values
    # (ncomp,nobs) @ (nobs,) -> (ncomp,)
    Vy <- crossprod(V, y)
    # square eta values
    # (ncomp,)^2 -> (ncomp,)
    Vy2 <- as.vector(Vy)^2
    # extract variance components
    # (2,)[i] -> scalar
    varE <- varComp[1]
    varU <- varComp[2]
    # calculate the ratio between genetic variance and error variance
    lambda <- varU / varE
    # calculate diagonal values for varU * G + varE * I matrix spectral decomposition
    dStar <- (d*lambda + 1)
    # calculate log-determinant using sum of logs of diagonal matrix
    sumLogD <- sum(log(dStar))
    # calculate log-likelihood in two parts
    neg2LogLik_1 <- (n*log(varE) + sumLogD )
    # calculate log-likelihood in two parts
    neg2LogLik_2 <- (sum(Vy2 / dStar)) / varE
    # calculate log-likelihood
    out <- neg2LogLik_1 + neg2LogLik_2
    # return log-likelihood
    return(out)
}

# @brief    Fit an ML estimate of variance components.
#
# @param    y
#           A vector of shape (nobs,) containing observed phenotypes.
# @param    EVD
#           A spectral decomposition of the genomic relationship matrix.
# @param    K
#           A genomic relationship matrix of shape (nobs,nobs). Must be symmetric.
#
# @return   out
#           A named list containing variance estiamtes and the log-likelihood.
fitML_v2 <- function(
    y,
    EVD = NULL,
    K = NULL
) {
    # if the eigenvalue decomposition of the genomic relationship matrix has not been provided
    if(is.null(EVD)) {
        # raise error if the genomic relationship matrix has not been provided
        if(is.null(K)) {
            stop('provide either K or its eigenvalue decomposition')
        }
        # otherwise take the spectral decomposition of the matrix, assuming matrix is symmetric.
        else {
            EVD = eigen(K, symmetric = TRUE)
        }
    }
    # calculate the variance of the observations
    # (nobs,) -> scalar
    varP = var(y)
    # construct an initial guess of the log-transformed variance components (varE,varU)
    # scalar * (2,) -> (2,)
    varComp = varP * c(.5,.5)
    # for the spectral decomposition, construct a mask for eigenvalues greater than approximately zero
    # (nobs,) -> (nobs,) 
    mask = EVD$values > 1e-5
    # extract eigenvalues that are greater than approximately zero
    # (nobs,) -> (ncomp,)
    d = EVD$values[mask]
    # extract eigenvectors for eigenvalues that are greater than approximately zero
    # (nobs,nobs) -> (nobs,ncomp)
    V = EVD$vectors[,mask]
    # use the Nelder-Mead method to minimize the -2logLik function for ML
    # fm is a list of optimization components
    fm = optim(
        par = varComp,
        fn = neg2LogLik_v2,
        V = V,
        d = d,
        y = y,
        n = length(y)
    )
    # extract estimated parameters
    # (2,)
    out = exp(fm$par)
    # extract heritability and log-likelihood
    # (4,)
    out = c(out, out[2] / sum(out), -2 * fm$value)
    # add names to the vector
    names(out) <- c('varE','varU','h2','log-lik')
    # return named vector
    return(out)
}

### Example
library(BGLR)
data(wheat)

# center and scale genotype matrix (nobs=599,nmkr=1279)
# (nobs,nmkr) -> (nobs,nmkr)
M = scale(wheat.X)

# take the transposed cross product: M @ M.T
# (nobs,nmkr) @ (nmkr,nobs) -> (nobs,nobs)
G = tcrossprod(M)

# calculate the mean of the diagonal and compute its inverse
# we take the inverse because in the next step multiplication is faster than division
# 1 / scalar -> scalar
k = 1 / mean(diag(G))

# scale the matrix
# scalar * (nobs,nobs) -> (nobs,nobs)
G = k * G

# take the spectral decomposition of the (symmetric) G matrix.
# (nobs,nobs) -> eigenvalues = (nobs,), eigenvectors = (nobs,nobs)
EVD = eigen(G, symmetric = TRUE)

# get the second y trait
y = wheat.Y[,2]

# get ML estimates of the variance components
fmML = fitML_v2(y=y,EVD=EVD) 
