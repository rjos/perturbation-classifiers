# Perturbation Classifiers

Installation:
-------------

The package can be installed using the following command:

.. code-block:: bash
    
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

References:
-----------

.. [1] : Araújo, E.L., Cavalcanti, G.D.C. & Ren, T.I. Perturbation-based classifier. Soft Comput (2020).