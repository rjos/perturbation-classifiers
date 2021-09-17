Perturbation Classifiers
========

Perturbation Classifiers is an easy-to-use library focused on the implementation of the Perturbation-based Classifier (PerC) [1]_ and subconcept Perturbation-based Classifier (sPerC). The library is is based on scikit-learn_, using the same method signatures: **fit**, **predict**, **predict_proba** and **score**.

Installation:
-------------

The package can be installed using the following command:

.. code-block:: bash
    
    # Clone repository
    git clone https://github.com/rjos/perturbation-classifiers.git
    cd perturbation-classifiers/
    
    # Installation lib
    python setup.py install


Dependencies:
-------------

perturbation_classifiers is tested to work with Python 3.7. The dependencies requirements are:

* scikit-learn(>=0.24.2)
* numpy(>=1.21.2)
* scipy(>=1.7.1)
* matplotlib(>=3.4.3)
* pandas(>=1.3.2)
* gap-stat(>=2.0.1)
* gapstat-rs(>=2.0.1)

These dependencies are automatically installed using the command above.

Examples:
---------

Here we show an example using the PerC method:

.. code-block:: python
    
    from perturbation_classifiers import PerC

    # Train a PerC model
    perc = PerC()
    perc.fit(X_train, y_train)

    # Predict new examples
    perc.predict(X_test)

and here we show an example using the sPerC method:

.. code-block:: python

    from perturbation_classifiers.subconcept import sPerC

    # Train a sPerC model
    sperc = sPerC()
    sperc.fit(X_train, y_train)

    # Predict new examples
    sperc.predict(X_test)

References:
-----------

.. [1] : Araújo, E.L., Cavalcanti, G.D.C. & Ren, T.I. Perturbation-based classifier. Soft Comput (2020).

.. _scikit-learn: http://scikit-learn.org/stable/