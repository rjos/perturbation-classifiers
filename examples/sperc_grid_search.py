# coding=utf-8

# Author: Rodolfo J. O. Soares <rodolfoj.soares@gmail.com>
"""
====================================================================
GridSearch sPerC models from a dataset
====================================================================

In this example we show how to apply grid search to sPerC model for a
classification dataset.

"""
from perturbation_classifiers.subconcept import sPerC

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split

import numpy as np

# Setting up the random state to have consistent results
rng = np.random.RandomState(42)

# Load a classification dataset
data = load_breast_cancer()
X = data.data
y = data.target

# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rng)

# Scale the variables between [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# sPerC parameters
parameters = {
    "aggregation_rule": ["min", "nearest_cluster"],
    "cluster_validation": ["gap", "sil"]
}

# Grid Search execution
grid_search = GridSearchCV(sPerC(), param_grid=parameters, cv=10, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model
best_clf = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate sPerC model
print('Evaluating technique:')
print('Classification accuracy sPerC: ', best_clf.score(X_test, y_test))
print('\tBest parameters: ', best_params)