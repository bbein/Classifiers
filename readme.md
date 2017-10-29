# Classifiers
Classifiers is a collection of mashine learning classifiers implemented in python.
The purpose of this package is to help me learn more about classifiers, unit-testing, 
logging, git, git-flow and best practices for all of them.

# Install

Run this script to install the package using:
    `pip install .`

To install the package for local development use:
    `pip install -e .`

To run all the tests for the classifiers use:
    `python setup.py test`

# Usage

```python
import logging
from Classifiers import LogisticRegression
model = LogisticRegression(max_iteration=20001, step_size=1e1, min_change=1e-6)
logging.basicConfig(level=logging.DEBUG) # to get more information about the training
feature_matrix = [[0., -1., 3.], [0., 0., 1.], [1., 0., 0.5]]
targets = [0, 0, 1]
print("weights: ", model.train(feature_matrix, targets))
print("predictions: ", model.predict_probability(feature_matrix))
```

this will produce the following output:

```
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 0: log likelihood = -0.0788934681329 change of log likelihood = inf
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 1: log likelihood = -0.0788906397081 change of log likelihood = 2.82842480367e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 2: log likelihood = -0.0788878122793 change of log likelihood = 2.82742876956e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 3: log likelihood = -0.0788849855368 change of log likelihood = 2.82674257823e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 4: log likelihood = -0.0788821593352 change of log likelihood = 2.82620155079e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 5: log likelihood = -0.0788793336066 change of log likelihood = 2.82572856114e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 6: log likelihood = -0.0788765083192 change of log likelihood = 2.82528749684e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 7: log likelihood = -0.0788736834577 change of log likelihood = 2.8248614401e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 8: log likelihood = -0.0788708590152 change of log likelihood = 2.82444246635e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 9: log likelihood = -0.0788680349884 change of log likelihood = 2.82402687124e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 10: log likelihood = -0.0788652113755 change of log likelihood = 2.82361291082e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 11: log likelihood = -0.0788623881757 change of log likelihood = 2.82319977463e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 12: log likelihood = -0.0788595653886 change of log likelihood = 2.82278708119e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 13: log likelihood = -0.078856743014 change of log likelihood = 2.82237465066e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 14: log likelihood = -0.0788539210516 change of log likelihood = 2.82196239798e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 15: log likelihood = -0.0788510995013 change of log likelihood = 2.82155028786e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 20: log likelihood = -0.0788369979274 change of log likelihood = 2.81949144831e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 30: log likelihood = -0.078808825625 change of log likelihood = 2.81538174418e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 40: log likelihood = -0.0787806943617 change of log likelihood = 2.81128257162e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 50: log likelihood = -0.0787526040324 change of log likelihood = 2.80719389067e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 60: log likelihood = -0.0787245545323 change of log likelihood = 2.80311566447e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 70: log likelihood = -0.0786965457572 change of log likelihood = 2.79904785438e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 80: log likelihood = -0.0786685776031 change of log likelihood = 2.79499042044e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 90: log likelihood = -0.0786406499664 change of log likelihood = 2.79094332667e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 100: log likelihood = -0.078612762744 change of log likelihood = 2.78690653621e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 200: log likelihood = -0.0783360908676 change of log likelihood = 2.74709707071e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 300: log likelihood = -0.0780633496436 change of log likelihood = 2.70827763371e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 400: log likelihood = -0.0777944418451 change of log likelihood = 2.67041333046e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 500: log likelihood = -0.0775292736538 change of log likelihood = 2.63347084672e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 600: log likelihood = -0.0772677545065 change of log likelihood = 2.59741836883e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 700: log likelihood = -0.077009796949 change of log likelihood = 2.56222550155e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 800: log likelihood = -0.076755316499 change of log likelihood = 2.52786318011e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 900: log likelihood = -0.0765042315156 change of log likelihood = 2.49430360988e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 1000: log likelihood = -0.0762564630764 change of log likelihood = 2.46152019834e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 2000: log likelihood = -0.0739455528172 change of log likelihood = 2.17126559798e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 3000: log likelihood = -0.0718958485314 change of log likelihood = 1.93626777056e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 4000: log likelihood = -0.0700594982383 change of log likelihood = 1.74260748409e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 5000: log likelihood = -0.0684002878297 change of log likelihood = 1.58060751465e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 6000: log likelihood = -0.0668902070743 change of log likelihood = 1.44334519137e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 7000: log likelihood = -0.0655071843071 change of log likelihood = 1.32574634248e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 8000: log likelihood = -0.0642335455198 change of log likelihood = 1.22401270319e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 9000: log likelihood = -0.0630549381585 change of log likelihood = 1.13524846901e-06
DEBUG:Classifiers.LogisticRegressionClassifier:iteration 10000: log likelihood = -0.0619595620073 change of log likelihood = 1.05720996268e-06
INFO:Classifiers.LogisticRegressionClassifier:stoped because the log likelihood change was to small iteration 10820: log likelihood = -0.0611165282834 change of log  likelihood = 9.99972894711e-07
weights: [7180.344663389312, 5.000000135316979, -2.764287490312489]
predictions: [1.6864846355561232e-06, 0.05928479985194321, 1.0]
```