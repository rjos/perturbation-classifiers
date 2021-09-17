# coding=utf-8

# Author: Rodolfo J. O. Soares <rodolfoj.soares@gmail.com>

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def min_rule(perturbations, cluster2class):
    """[summary]

    Parameters
    ----------
    perturbations : [description]
    
    cluster2class : [description]

    Returns
    -------
    [type]: [description]
    """
    classes = np.array(list(cluster2class.values()))

    perturbations_min = np.full((perturbations.shape[0], len(set(classes))), np.inf)

    for i, c in enumerate(set(classes)):
        perturbations_min[:, i] = np.min(perturbations[:, (classes == c)], axis=1)

    return perturbations_min

def avg_rule(perturbations, cluster2class):
    """[summary]

    Parameters
    ----------
    perturbations : [description]
        
    cluster2class : [description]

    Returns
    -------
    [type]: [description]
    """
    classes = np.array(list(cluster2class.values()))

    perturbations_mean = np.full((perturbations.shape[0], len(set(classes))), np.inf)

    for i, c in enumerate(set(classes)):
        perturbations_mean[:, i] = np.mean(perturbations[:, (classes == c)], axis=1)

    return perturbations_mean

def median_rule(perturbations, cluster2class):
    """[summary]

    Parameters
    ----------
    perturbations : [description]

    cluster2class : [description]

    Returns
    -------
    [type]: [description]
    """
    classes = np.array(list(cluster2class.values()))

    perturbations_median = np.full((perturbations.shape[0], len(set(classes))), np.inf)

    for i, c in enumerate(set(classes)):
        perturbations_median[:, i] = np.median(perturbations[:, (classes == c)], axis=1)

    return perturbations_median

def nearest_cluster_rule(X, perturbations, cluster2class, centroids):
    """[summary]

    Parameters
    ----------
    X : [description]

    perturbations : [description]

    cluster2class : [description]
    
    centroids : [description]

    Returns
    -------
    [type]: [description]
    """
    classes = np.array(list(cluster2class.values()))

    perturbations_nearest_cluster = np.full((perturbations.shape[0], len(set(classes))), np.inf)

    for i, c in enumerate(set(classes)):
        class_centers_ = centroids[(classes == c), :]

        dists = euclidean_distances(X, class_centers_)
        nearest_clusters_idx = dists.argmin(axis=1)

        perturbation_clusters = perturbations[:, (classes == c)]
        for j, p in enumerate(nearest_clusters_idx):
            perturbations_nearest_cluster[j, i] = perturbation_clusters[j, p]

    return perturbations_nearest_cluster
