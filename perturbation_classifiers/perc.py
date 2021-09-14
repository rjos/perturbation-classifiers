# coding=utf-8

# Author: Rodolfo José de Oliveira Soares <rodolfoj.soares@gmail.com>

import numpy as np

from perturbation_classifiers.base import BasePerC
from perturbation_classifiers.estimation import estimate_mean_vector_per_class, estimate_covariance_matrix_per_class, estimate_delta_mean_vector_per_class, estimate_delta_covariance_matrix_per_class
from perturbation_classifiers.perturbation import perturbation_mean, perturbation_covariance, perturbation_combination

class PerC(BasePerC):

    def __init__(self, mode="auto"):
        super(PerC, self).__init__(mode=mode)
    
    def fit(self, X, y):
        """[summary]

        Parameters
        ----------
        X : [description]
        
        y : [description]

        Returns
        -------
        [type]: [description]
        """
        super(PerC, self).fit(X, y)

        # Store the classes and its quantity seen during fit
        self.classes_, self.count_classes_ = np.unique(y, return_counts=True)

        # Check that mode paramenters have correct value
        self._validate_parameters()

        # Compute means vectors for each class
        self.means_ = estimate_mean_vector_per_class(X, y, self.classes_, self.count_classes_)

        # Compute covariances matrix for each class
        if self.mode != "mean":
            self.covariances_ = estimate_covariance_matrix_per_class(X, y, self.classes_, self.count_classes_, self.means_)

        # Compute pseudo-inverse matrix for each covariance matrix
        if self.mode == "auto":
            self.inverse_covariances_ = np.linalg.pinv(self.covariances_)

        return self
    
    def perturbation(self, X):
        """[summary]

        Parameters
        ----------
        X : [description]

        Returns
        -------
        [type]: [description]
        """
        super(PerC, self).perturbation(X)

        # Compute perturbation-based the mode input
        perturbations = None

        if self.mode == "mean": 
            # Compute the perturbation-based only mean vector
            delta_mean_vectors = estimate_delta_mean_vector_per_class(X, self.means_, self.count_classes_)
            perturbations = perturbation_mean(delta_mean_vectors)
        elif self.mode == "covariance": 
            # Compute the perturbation-based only covariance matrix
            delta_covariances_matrix = estimate_delta_covariance_matrix_per_class(X, self.means_, self.covariances_, self.count_classes_)
            perturbations = perturbation_covariance(delta_covariances_matrix)
        else: 
            # Compute the perturbation-based combination mean vector and covariance matrix
            delta_mean_vectors = estimate_delta_mean_vector_per_class(X, self.means_, self.count_classes_)
            delta_covariances_matrix = estimate_delta_covariance_matrix_per_class(X, self.means_, self.covariances_, self.count_classes_)
            perturbations = perturbation_combination(X, self.means_, self.inverse_covariances_, delta_mean_vectors, delta_covariances_matrix)

        return perturbations
    
    def _validate_parameters(self):
        """[summary]

        Raises
        ------
        ValueError: [description]
        """
        # Validate mode parameter
        if self.mode not in ["auto", "mean", "covariance"]:
            raise ValueError(
                'Invalid value for parameter "mode".'
                ' "mode" should be one of these options '
                '"auto", "mean", "covariance"'
            )