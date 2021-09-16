# coding=utf-8

# Author: Rodolfo J. O. Soares <rodolfoj.soares@gmail.com>

import numpy as np
from abc import abstractmethod, ABCMeta

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

from perturbation_classifiers.util.prob_function import softmin

class BasePerC(BaseEstimator, ClassifierMixin):
    """[summary]
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, mode="auto"):
        self.mode = mode
    
    @abstractmethod
    def fit(self, X, y):
        """[summary]

        Parameters
        ----------
        X : [description]
            
        y : [description]

        Returns
        -------
        self
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
    
    def predict(self, X):
        """[summary]

        Parameters
        ----------
        X : [description]

        Returns
        -------
        y : [description]
        """
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)

        # Compute the perturbation of each class
        perturbations = self.perturbation(X)

        # Predict the class label for each sample query
        idx = perturbations.argmin(axis=1)
        return self.classes_[idx]
    
    @abstractmethod
    def perturbation(self, X):
        """[summary]

        Parameters
        ----------
        X : [description]

        Returns
        -------
        perturbations : 
        """
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
    
    def score(self, X, y):
        """[summary]

        Parameters
        ----------
        X : [description]
        y : [description]

        Returns
        -------
        [type]: [description]
        """

        # Compute the prediction of X set
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def predict_proba(self, X):
        """[summary]

        Parameters
        ----------
        X : [description]

        Returns
        -------
        [type]: [description]
        """
        # Compute perturbation of each class
        perturbations = self.perturbation(X)
        return softmin(perturbations)
    
    @abstractmethod
    def __validate_parameters(self):
        """[summary]
        """
        pass