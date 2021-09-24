# coding=utf-8

# Author: Rodolfo J. O. Soares <rodolfoj.soares@gmail.com>
"""
====================================================================
Train PerC model from keel dataset format and using a 
custom scorer to evaluate the model
====================================================================

In this example we show how to apply PerC models for a
classification dataset and evaluate it using a custom scorer.

"""
from numpy.core.numeric import cross
from perturbation_classifiers.util.dataset import load_keel_file
from perturbation_classifiers import PerC

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline

import os
import numpy as np

# Setting up the random state to have consistent results
rng = np.random.RandomState(42)

# Load a classification dataset
dataset_name = 'balance.dat'
dataset_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', dataset_name)
keel_dataset = load_keel_file(dataset_path)

X, y = keel_dataset.get_data()

# Encoder the targets
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Define a custom scorer
def custom_auc(y_true, y_pred):
    lb = LabelBinarizer()

    y_true_lb = lb.fit_transform(y_true)
    y_pred_lb = lb.transform(y_pred)

    return roc_auc_score(y_true_lb, y_pred_lb)

custom_scoring = {
    "custom_auc": make_scorer(custom_auc)
}

# Define a Pipeline
pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("perc", PerC())
    ]
)

# Evaluate PerC models
results = cross_validate(pipe, X, y, cv=10, scoring=custom_scoring)

print('Evaluating PerC techniques:')
print('Classification ROC AUC PerC: ', np.mean(results["test_custom_auc"]))