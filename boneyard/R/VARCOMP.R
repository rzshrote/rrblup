#!/usr/bin/env Rscript
# R implementation of ML and REML for a very simple LMM with a single random effect and a single intercept fixed effect.

# @brief    -2 log-likelihood function for ML estimation
# 
# @param    logVar
#           A vector of shape (2,) containing parameter estimates (varE,varU) in the log scale.
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
neg2LogLik <- function(
    logVar,
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
    # exponentiate the log-variance components
    # scalar -> scalar
    varE <- exp(logVar[1])
    varU <- exp(logVar[2])
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
fitML <- function(
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
    logVar = varP * c(.5,.5)
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
        par = logVar,
        fn = neg2LogLik,
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

neg2_REML <- function(
    y,
    Xt,
    vars,
    d,
    n,
    nSuccess = 1,
    nFailure = 1
) {
    ###
    # y=U'eHat where is eHat=OLS residuals
    # Xt=X'U
    # vars=c(varE,varU)
    # d: eigenvalues
    # n: length(y)
    # objective function is -2*L2 (page 1 in http://www.aps.uoguelph.ca/~lrs/Animalz/lesson11/lesson11.pdf)
    # gustavoc@msu.edu 11/30/2016
    ###
    varE = (vars[1])
    varU = (vars[2])
    w = d * varU + varE
    neg2_reml_1 = sum(log(w))
    Xt = scale(Xt, center=FALSE, scale = 1/sqrt(w))
    C = tcrossprod(Xt)
    neg2_reml_2 = log(det(C))
    neg2_reml_3 = sum((y^2)/w)
    h2 = varU / (varU+varE)
    neg2LogPrior = -2* ((nSuccess-1) * log(h2) + (nFailure-1) * log(1-h2))
    out = neg2_reml_1+neg2_reml_2+neg2_reml_3+neg2LogPrior
    return(out)
}

fitREML <- function(
    y,
    EVD = NULL,
    K = NULL,
    n = length(y),
    X = matrix(nrow = n, ncol = 1, 1),
    nSuccess = 1,
    nFailure = 1,
    computeHessian = FALSE
) {
    ###
    # y: phenotype
    # EVD = eigen(G)
    # X: incidence matrix for fixed effects
    # gustavoc@msu.edu 11/30/2016
    ###
    if(is.null(EVD)) {
        if(is.null(K)) {
            stop('provide either K or its eigenvalue decomposition')
        }
        else {
            EVD=eigen(K)
        }
    }
    varY = var(y)
    tmp = rep(TRUE,length(EVD$values)) # EVD$values>1e-5
    d = EVD$values[tmp]
    U = EVD$vectors[,tmp]
    eHat = residuals(lsfit(y=y,x=X,intercept=F))
    y = crossprod(U,eHat)
    Xt = crossprod(X,U)
    fm = bobyqa(
        par = varY * rep(.5,2),
        fn = neg2_REML,
        lower = rep(1e-5, 2) * varY,
        upper = rep(1.5, 1.5) * varY,
        y = y,
        Xt = Xt,
        d = d,
        n = n,
        nSuccess = nSuccess,
        nFailure = nFailure
    )
    estimates = c(
        fm$par,
        fm$par[2]/sum(fm$par)
    )
    names(estimates) <- c('varE','varU','h2')
    out = list(
        estimates = estimates,
        logREML = -2 * fm$fval
    )
    if(computeHessian) {
        COV1=solve(
            hessian(
                fun = neg2_REML,
                x = fm$par,
                y = y,
                Xt = Xt,
                d = d,
                n = n,
                nSuccess = nSuccess,
                nFailure = nFailure
            )
        )
        X = matrix(nrow=100000, ncol=2, rnorm(200000)) %*% chol(COV1)
        X[,1] = X[,1] + estimates[1]
        X[,2] = X[,2] + estimates[2]
        COV = matrix(nrow=3,ncol=3,NA)
        COV[1:2,1:2] = COV1
        h2 = X[,2] / rowSums(X)
        COV[3,3] = var(h2)
        COV[3,1] = COV[1,3] = cov(X[,1], h2)
        COV[3,2] = COV[2,3] = cov(X[,2], h2)
        out$SEs = sqrt(diag(COV))
        out$vcov = COV
    }
    return(out)
}

### Example
library(minqa)
library(numDeriv)
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

# get REML estimates of the variance components
fmREML = fitREML(y=y,EVD=EVD)

# get ML estimates of the variance components
fmML = fitML(y=y,EVD=EVD) 
