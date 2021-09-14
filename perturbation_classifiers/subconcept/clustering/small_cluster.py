# coding=utf-8

# Author: Rodolfo José de Oliveira Soares <rodolfoj.soares@gmail.com>

import numpy as np

def find_small_clusters(labels, alpha):
    """[summary]

    Parameters
    ----------
    labels ([type]): [description]
    alpha ([type]): [description]

    Returns
    -------
    [type]: [description]
    """

    # Get unique clusters labels and its counts
    clusters, clusters_sizes = np.unique(labels, return_counts=True)
    # Ordered clusters label by size in descending way
    reversed_order_idx = clusters_sizes.argsort()[::-1]

    # Compute the threshold size of train set 
    threshold_size = np.sum(clusters_sizes) * (1 - alpha)

    # Find the small clusters idx
    small_clusters_idx = [clusters[reversed_order_idx[i]] for i in range(len(clusters)) if np.sum(clusters_sizes[reversed_order_idx[:i]]) > threshold_size]
    return small_clusters_idx