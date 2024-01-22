#!/usr/bin/env Rscript

#       varE         varU           h2      log-lik 
#   0.5725375    0.4690475    0.4503209 -960.2798117 
#       varE        varU          h2     log-lik 
#   1.772578    1.598278    0.474146 -960.279804 

library(BGLR)
data(wheat)

# get data
M = wheat.X
y = wheat.Y[,2]

# create genomic relationship matrix
# (nobs,nobs)
M = scale(M)
G = tcrossprod(M)
k = 1.0 / mean(diag(G))
G = k * G

# take the spectral decomposition of the (symmetric) G matrix.
# (nobs,nobs) -> eigenvalues = (nobs,), eigenvectors = (nobs,nobs)
EVD = eigen(G, symmetric = TRUE)

# center trait data around mean
# (nobs,)
y = y - mean(y)

# for the spectral decomposition, construct a mask for eigenvalues greater than approximately zero
# (nobs,) -> (nobs,) 
mask = EVD$values > 1e-5

# extract eigenvalues that are greater than approximately zero
# (nobs,) -> (ncomp,)
d = EVD$values[mask]

# extract eigenvectors for eigenvalues that are greater than approximately zero
# (nobs,nobs) -> (nobs,ncomp)
V = EVD$vectors[,mask]

# calculate V.T @ y to get eta values
# (ncomp,nobs) @ (nobs,) -> (ncomp,)
eta = crossprod(V, y)

# square eta values
# (ncomp,)^2 -> (ncomp,)
etasq = as.vector(eta)^2

# variance components
varE = 0.5725375
varU = 0.4690475

# calculate the ratio between genetic variance and error variance
lambda = varU / varE

# calculate diagonal values for varU * G + varE * I matrix spectral decomposition
dStar = (d*lambda + 1)

# calculate log-determinant using sum of logs of diagonal matrix
sumLogD = sum(log(dStar))

# calculate log-likelihood in two parts
neg2LogLik_1 = (length(y) * log(varE) + sumLogD )

# calculate log-likelihood in two parts
neg2LogLik_2 = (sum(etasq / dStar)) / varE

# calculate log-likelihood
n2ll = neg2LogLik_1 + neg2LogLik_2

h2 = varU / (varU + varE)

ll = -2.0 * n2ll
