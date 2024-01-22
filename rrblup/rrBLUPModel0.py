"""
Module containing RR-BLUP genomic prediction implementation classes. 
"""

from numbers import Integral
from typing import Optional
from rrblup.GenomicPredictionModel import GenomicPredictionModel
import numpy

class rrBLUPModel0(GenomicPredictionModel):
    """
    RR-BLUP model for fitting a single random effect and a single intercept fixed effect.

    If the observations are mean centered, then
    """

    ########################## Special Object Methods ##########################
    def __init__(
            self,
            trait: Optional[numpy.ndarray],
            method: str = "ML",
        ) -> None:
        """
        Constructor for RRBLUPModel.
        
        Parameters
        ----------
        trait : numpy.ndarray, None
            Names of traits.
        method : str
            Fitting method to use. Options are ``{"ML"}``.
        """
        # assignments
        self.trait = trait

    ############################ Object Properties #############################

    ############### Genomic Model Parameters ###############
    @property
    def nexplan(self) -> Integral:
        """Number of explanatory variables required by the model."""
        return self._nexplan

    @property
    def nparam(self) -> Integral:
        """Number of model parameters."""
        raise NotImplementedError("property is abstract")

    ################## Genomic Model Data ##################
    @property
    def trait(self) -> numpy.ndarray:
        """Names of the traits predicted by the model."""
        raise NotImplementedError("property is abstract")
    @trait.setter
    def trait(self, value: numpy.ndarray) -> None:
        """Set the names of the traits predicted by the model"""
        if not isinstance(value, numpy.ndarray):
            raise TypeError("'trait' must be of type ``numpy.ndarray``, but received type ``{0}``".format(type(value).__name__))
        if value.ndim != 1:
            raise ValueError("numpy.ndarray 'trait' must have ndim == 1, but received ndim == {0}".format(value.ndim))
        self._trait = value

    @property
    def ntrait(self) -> int:
        """Number of traits predicted by the model."""
        return len(self.ntrait)

    ############################## Object Methods ##############################

    #################### Model copying #####################
    def copy(
            self
        ) -> 'GenomicPredictionModel':
        """
        Make a shallow copy of the GenomicPredictionModel.

        Returns
        -------
        out : GenomicPredictionModel
            A shallow copy of the original GenomicPredictionModel
        """
        raise NotImplementedError("method is abstract")

    def deepcopy(
            self,
            memo: Optional[dict]
        ) -> 'GenomicPredictionModel':
        """
        Make a deep copy of the GenomicPredictionModel.

        Parameters
        ----------
        memo : dict
            Dictionary of memo metadata.

        Returns
        -------
        out : GenomicPredictionModel
            A deep copy of the original GenomicPredictionModel
        """
        raise NotImplementedError("method is abstract")

    ####### methods for model fitting and prediction #######
    def fit(
            self, 
            Y: numpy.ndarray, 
            X: numpy.ndarray, 
            Z: numpy.ndarray, 
            **kwargs: dict
        ) -> None:
        """
        Fit the model.

        Parameters
        ----------
        Y : numpy.ndarray
            A phenotype matrix of shape (n,t).
        X : numpy.ndarray
            A covariate matrix of shape (n,q) containing predictors for fixed effects.
        Z : numpy.ndarray
            A genotypes matrix of shape (n,p) containing predictors for random effects.
        kwargs : dict
            Additional keyword arguments.
        """
        raise NotImplementedError("method is abstract")

    def predict(
            self, 
            X: numpy.ndarray, 
            Z: numpy.ndarray, 
            **kwargs: dict
        ) -> numpy.ndarray:
        """
        Predict phenotypic values.

        Parameters
        ----------
        X : numpy.ndarray
            A matrix of covariates.
        Z : numpy.ndarray
            A matrix of genotype values.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Y_hat : numpy.ndarray
            A matrix of predicted phenotypic values.
        """
        raise NotImplementedError("method is abstract")

    def score(
            self, 
            Y: numpy.ndarray, 
            X: numpy.ndarray, 
            Z: numpy.ndarray, 
            **kwargs: dict
        ) -> numpy.ndarray:
        """
        Return the coefficient of determination R**2 of the prediction.

        Parameters
        ----------
        Y : numpy.ndarray
            A matrix of phenotypes.
        X : numpy.ndarray
            A matrix of covariates.
        Z : numpy.ndarray
            A matrix of genotypes.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Rsq : numpy.ndarray
            A coefficient of determination array of shape ``(t,)``.

            Where:

            - ``t`` is the number of traits.
        """
        raise NotImplementedError("method is abstract")

    ######## methods for estimated breeding values #########
    def gebv(
            self, 
            Z: numpy.ndarray, 
            **kwargs: dict
        ) -> numpy.ndarray:
        """
        Calculate genomic estimated breeding values.

        Parameters
        ----------
        Z : numpy.ndarray
            A matrix of genotype values.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        gebv_hat : numpy.ndarray
            A matrix of genomic estimated breeding values.
        """
        raise NotImplementedError("method is abstract")

    ############################## Class Methods ###############################

    ############################## Static Methods ##############################
