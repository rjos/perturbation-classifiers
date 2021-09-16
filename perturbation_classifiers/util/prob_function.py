# coding=utf-8

# Author: Rodolfo J. O. Soares <rodolfoj.soares@gmail.com>

import numpy as np

def softmax(vector, theta=1.0):
    """[summary]

    Parameters
    ----------
    vector : [description]
        
    theta : [description]. Defaults to 1.0.

    Returns
    -------
    dist : [description]
    """
    
    w = np.atleast_2d(vector)
    e = np.exp(np.array(w) / theta)
    dist = e / np.sum(e, axis=1).reshape(-1, 1)
    return dist

def softmin(vector):
    """[summary]

    Parameters
    ----------
    vector ([type]): [description]

    Returns
    -------
    [type]: [description]
    """

    return softmax(-1 * vector)