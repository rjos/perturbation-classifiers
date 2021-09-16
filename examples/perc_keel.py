# coding=utf-8

# Author: Rodolfo J. O. Soares <rodolfoj.soares@gmail.com>
"""
====================================================================
Train PerC models from keel dataset format
====================================================================

In this example we show how to apply PerC model for a
classification dataset.

"""
from perturbation_classifiers.util.dataset import load_keel_file
from perturbation_classifiers import PerC

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import os
import numpy as np

# Setting up the random state to have consistent results
rng = np.random.RandomState(42)


# Load a classification dataset
dataset_name = 'balance.dat'
dataset_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', dataset_name)
keel_dataset = load_keel_file(dataset_path)

X, y = keel_dataset.get_data()

# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rng)

# Scale the variables between [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encoder the targets
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Train PerC models
perc_mean = PerC(mode="mean")
perc_mean.fit(X_train, y_train)

perc_cov = PerC(mode="covariance")
perc_cov.fit(X_train, y_train)

perc_comb = PerC()
perc_comb.fit(X_train, y_train)

# Evaluate PerC models
print('Evaluating PerC techniques:')
print('Classification accuracy PerC<Mean>: ', perc_mean.score(X_test, y_test))
print('Classification accuracy PerC<Cov>: ', perc_cov.score(X_test, y_test))
print('Classification accuracy PerC<Comb>: ', perc_comb.score(X_test, y_test))