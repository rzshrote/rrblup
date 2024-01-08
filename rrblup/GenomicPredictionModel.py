"""
Module defining abstract interface for genomic prediction models.
"""

from abc import ABCMeta, abstractmethod
from numbers import Integral
from typing import Optional
import numpy


class GenomicPredictionModel(metaclass=ABCMeta):
    """
    Abstract class defining genomic prediction model objects.
    """

    ########################## Special Object Methods ##########################
    @abstractmethod
    def __copy__(
            self
        ) -> 'GenomicPredictionModel':
        """
        Make a shallow copy of the ``GenomicPredictionModel``.

        Returns
        -------
        out : GenomicPredictionModel
            A shallow copy of the ``GenomicPredictionModel``.
        """
        raise NotImplementedError("method is abstract")

    @abstractmethod
    def __deepcopy__(
            self, 
            memo: Optional[dict]
        ) -> 'GenomicPredictionModel':
        """
        Make a deep copy of the ``GenomicPredictionModel``.

        Parameters
        ----------
        memo : dict
            Dictionary of memo metadata.

        Returns
        -------
        out : GenomicPredictionModel
            A deep copy of the ``GenomicPredictionModel``.
        """
        raise NotImplementedError("method is abstract")

    ############################ Object Properties #############################

    ############### Genomic Model Parameters ###############
    @property
    @abstractmethod
    def nexplan(self) -> Integral:
        """Number of explanatory variables required by the model."""
        raise NotImplementedError("property is abstract")

    @property
    @abstractmethod
    def nparam(self) -> Integral:
        """Number of model parameters."""
        raise NotImplementedError("property is abstract")

    ################## Genomic Model Data ##################
    @property
    @abstractmethod
    def trait(self) -> numpy.ndarray:
        """Names of the traits predicted by the model."""
        raise NotImplementedError("property is abstract")
    @trait.setter
    @abstractmethod
    def trait(self, value: numpy.ndarray) -> None:
        """Set the names of the traits predicted by the model"""
        raise NotImplementedError("property is abstract")

    @property
    @abstractmethod
    def ntrait(self) -> int:
        """Number of traits predicted by the model."""
        raise NotImplementedError("property is abstract")
    @ntrait.setter
    @abstractmethod
    def ntrait(self, value: int) -> None:
        """Set the number of traits predicted by the model"""
        raise NotImplementedError("property is abstract")

    ############################## Object Methods ##############################

    #################### Model copying #####################
    @abstractmethod
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

    @abstractmethod
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
    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
    @abstractmethod
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
