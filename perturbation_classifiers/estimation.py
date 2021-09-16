# coding=utf-8

# Author: Rodolfo J. O. Soares <rodolfoj.soares@gmail.com>

import numpy as np

def estimate_mean_vector_per_class(X, y, unique_classes, count_classes):
    """[summary]

    Parameters
    ----------
    X : [description]

    y : [description]

    unique_classes : [description]

    count_classes : [description]

    Returns
    -------
    [type]: [description]
    """

    # Get the data belongs to each class
    instance_classes = list(
        map(lambda w: X[y == w], unique_classes)
    )

    # Compute the data mean of each class
    mean_vectors = list(
        map(lambda x, c: x.sum(axis=0) / c, instance_classes, count_classes)
    )
    
    return np.array(mean_vectors)   


def estimate_delta_mean_vector_per_class(X, mean_vectors, count_classes):
    """[summary]

    Parameters
    ----------
    X : [description]
    
    mean_vectors : [description]
    
    count_classes : [description]

    Returns
    -------
    [type]: [description]
    """

    # Compute the mean vector after simulation insert the sample query 
    delta_mean_vectors = list(
        map(lambda x: (x - mean_vectors) / (count_classes[:, np.newaxis] + 1), X)
    )

    return np.array(delta_mean_vectors)


def estimate_covariance_matrix_per_class(X, y, unique_classes, count_classes, mean_vectors):
    """[summary]

    Parameters
    ----------
    X : [description]
    
    y : [description]
    
    unique_classes : [description]
    
    count_classes : [description]
    
    mean_vectors : [description]

    Returns
    -------
    [type]: [description]
    """

    covariances_matrix = np.zeros((unique_classes.shape[0], X.shape[1], X.shape[1]), dtype=np.float_)
    for i, (w, c) in enumerate(zip(unique_classes, count_classes)):

        # Get data point belongs to each class
        data_per_class = X[y == w]

        # Get the mean of the class
        mean = mean_vectors[i]

        # Compute the covariance matrix belongs to each class
        data_mean_diff = (data_per_class - mean)
        data_covariance = np.dot(data_mean_diff.T, data_mean_diff) / c
        covariances_matrix[i] = data_covariance

    return np.asarray(covariances_matrix)


def estimate_delta_covariance_matrix_per_class(X, mean_vectors, covariances_matrix, count_classes):
    """[summary]

    Parameters
    ----------
    X : [description]
    
    mean_vectors : [description]
    
    covariances_matrix : [description]
    
    count_classes : [description]

    Returns
    -------
    [type]: [description]
    """

    # Compute difference between each test instance and each mean of the class
    X_diff_mean = list(
        map(lambda x: list(
            map(lambda m: (x - m).reshape(len(x), 1), mean_vectors)
        ), X)
    )

    # Compute the delta covariance matrix of each class
    delta_covariances_matrix = list(
        map(lambda perts: list(
            map(lambda per, n: np.dot(per, per.T) / (n + 1), perts, count_classes)
        ), X_diff_mean)
    )

    delta_covariances_matrix = list(
        map(lambda perts: list(
            map(lambda cov, per, n: ((-(cov / n)) + per), covariances_matrix, perts, count_classes)
        ), delta_covariances_matrix)
    )

    return np.array(delta_covariances_matrix)

