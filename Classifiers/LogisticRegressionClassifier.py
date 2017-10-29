"""
Logistic Regression Classifier module.
"""
# TODO: allow intercept training as a parameter
# TODO: perform gradiant assent in a function defined in the Core module and allow for other solvers
# TODO: implement rige/l2 regularization
# TODO: implement lasso/l1 regularization
# TODO: implement coardinate decent
# TODO: allow for selection of the solver
# TODO: implement dynamic step sizes
# TODO: implement automatic checks for convergence

import math
import logging

from Classifiers import Core

LOGGER = logging.getLogger(__name__)

class LogisticRegression(object):
    """
    Class to perform logistic regression and perform classification
    The model can be trained using the `.fit(feature_matrix, target)` function.
    After training:
    The model can perform classification using the `.predict(feature_matrix)` function.
    The model can return prpabilities using the `.predict_propabilities(feature_matrix)` function.
    """

    def __init__(self, step_size=1e-7, max_iteration=301, min_change=1e-8, cut_off=0.5):
        """
        Initilize the logistic regression model with `step_size`,
        `max_iteration`, `min_change` and `cut_off`

        # Parameters
        step_size: float, step size for the implizit gradient assend
        min_change: float, the minimal change of the log likelihood to perform another step
        max_iteration: int, maximum number of iteration during training
        cut_off: float, the cut_off value for when predictions are counted as 1 and 0
        """
        self._coefficients = None
        self.step_size = step_size
        self.max_iteration = max_iteration
        self.cut_off = cut_off
        self.min_change = min_change

    @property
    def coefficients(self):
        """
        Coefficients of the logistic regression model

        # Returns

        a copy of the private variable `_coefficients`
        """
        # return a shallow copy of coefficents
        return self._coefficients[:]

    def _compute_log_likelihood(self, feature_matrix, targets):
        """
        calculates the log likelihood to find out if the algorythem has converged

        # Parameters

        feature_matrix : list-like, e.g: [[x_11, x_12, ...], [x_21, x_22, ...], ...]
        targets : list-like, needs to be the same length as `feature_matrix` e.g: [0, 1, 0, ...]

        # Returns

        log_likelihood : float
        """
        scores = Core.dot_product_matrix_vector(feature_matrix, self.coefficients)
        log_exps = [math.log(1. + math.exp(-score)) for score in scores]
        # Simple check to prevent overflow
        for i, _ in enumerate(log_exps):
            if math.isinf(log_exps[i]):
                log_exps[i] = -scores[i]
        log_likelihood = 0
        for log_exp, target, score in zip(log_exps, targets, scores):
            log_likelihood += (target-1)*score - log_exp
        return log_likelihood

    def train(self, feature_matrix, targets, initial_coefficients=None):
        """
        Trains the model using the features in `feature_matrix` to predict the `targets`.
        For a smarter initialzation or a continuation of the traing `initial_coefficients`
        can be passed used.

        # Parameters

        feature_matrix : list-like, e.g: [[x_11, x_12, ...], [x_21, x_22, ...], ...]
        targets : list-like, needs to be the same length as `feature_matrix` e.g: [0, 1, 0, ...]
        initial_coefficients : list-like, the length needs to be the same as the number of features

        # Example

        >>> model = LogisticRegression()
        >>> feature_matrix = [[0., -1., 3.], [0., 0., 1.], [1., 0., 0.5]]
        >>> targets = [0, 0, 1]
        >>> model.train(feature_matrix, targets)
        [6.0199336864288055e-05, 1.5049788360983397e-05, -7.524964726549306e-06]
        >>> model.predict(feature_matrixty)
        [0, 0, 1]
        >>> model.predict_probability(feature_matrix)
        [0.4999905938293659, 0.49999811875881833, 0.5000141092136214]
        """
        # TODO: Further refactor and add more unit tests
        # test inputs
        if len(feature_matrix) != len(targets):
            message = "The length of the target has to be the same as the feature matrix!"
            message += "Length vector: {0}. ".format(len(targets))
            message += "Length matrix row 0: {0}".format(len(feature_matrix))
            raise ValueError(message)
        for i, row in enumerate(feature_matrix):
            if len(feature_matrix[0]) != len(row):
                message = "The length of each row in the feature matrix has to be the same!"
                message += "Length feature matrix row {0}: {1}. ".format(i, len(row))
                message += "Length feature matrix row 0: {0}".format(len(feature_matrix[0]))
                raise ValueError(message)
        for i, target in enumerate(targets):
            if target != 1 and target != 0:
                message = "The target needs to contain only of 1 and 0!"
                message += "Target row {0}: {1}. ".format(i, target)
                raise ValueError(message)
        # initilize coefficients
        if not initial_coefficients:
            initial_coefficients = [0]*len(feature_matrix[0])
        self._coefficients = initial_coefficients
        last_log_likelihood = float('inf')
        # Train model
        for iteration in xrange(self.max_iteration):
            predictions = self.predict_probability(feature_matrix)
            errors = [target - prediction for target, prediction in zip(targets, predictions)]
            # Update coffeicents w(i+1) = w(i) + step_size*derivative
            for j in xrange(len(self._coefficients)): # loop over each coefficient
                derivative = Core.dot_product_vector_vector(errors, feature_matrix[j])
                self._coefficients[j] = self._coefficients[j] + self.step_size * derivative
            log_likelihood = self._compute_log_likelihood(feature_matrix, targets)
            change_log_likelihood = abs(last_log_likelihood-log_likelihood)
            last_log_likelihood = log_likelihood
            # Log if the training ends because the change in loglikelihood was to small
            if change_log_likelihood < self.min_change:
                message = "stoped because the log likelihood change was to small"
                message += ' iteration {0}: log likelihood = {1}'.format(iteration, log_likelihood)
                message += ' change of log  likelihood = {0}'.format(change_log_likelihood)
                LOGGER.info(message)
                break
            # Log intermediat steps
            if (iteration <= 15 or (iteration <= 100 and iteration % 10 == 0)
                    or (iteration <= 1000 and iteration % 100 == 0)
                    or (iteration <= 10000 and iteration % 1000 == 0)):
                message = 'iteration {0}: log likelihood = {1}'.format(iteration, log_likelihood)
                message += " change of log likelihood = {0}".format(change_log_likelihood)
                LOGGER.debug(message)
            # Log if the training ended because we reachd the last iteration
            if iteration+1 == self.max_iteration:
                message = "stoped because maximum numbers of iterations was reached: "
                message += "{0}".format(self.max_iteration)
                LOGGER.info(message)
        return self.coefficients

    def predict_probability(self, feature_matrix):
        """
        produces probablistic estimate for P(y_i = +1 | x_i, w).
        probabilities ranges between 0 and 1.

        # Parameters

        feature_matrix : list-like, e.g: [[x_11, x_12, ...], [x_21, x_22, ...], ...]

        # Returns

        list of probabilities between 0 and 1

        # Example:

        >>> model = LogisticRegression()
        >>> model._coefficients = [1., 3., -1.]
        >>> feature_matrix = [[1., 2., 3.], [1., -1., -1], [0., 0., 0.]]
        >>> model.predict_probability(feature_matrix, coefficients)
        [0.9820137900379085, 0.2689414213699951, 0.5]
        """
        scores = Core.dot_product_matrix_vector(feature_matrix, self.coefficients)
        # Compute P(y_i = +1 | x_i, w)
        probabilities = [1.0 / (1 + math.exp(-score)) for score in scores]
        return probabilities

    def predict(self, feature_matrix):
        """
        produces classification result of 0 or 1.
        if P(y_i = +1 | x_i, w) > cut_off: return 1
        if P(y_i = +1 | x_i, w) <= cut_off: return 0

        # Parameters

        feature_matrix : list-like, e.g: [[x_11, x_12, ...], [x_21, x_22, ...], ...]

        # Returns

        list, of int either 0 ot 1

        # Example:

        >>> model = LogisticRegression()
        >>> model._coefficients = [1., 3., -1.]
        >>> feature_matrix = [[1., 2., 3.], [1., -1., -1], [0., 0., 0.]]
        >>> model.predict(feature_matrix)
        [1, 0, 0]
        """
        classifications = []
        for propability in self.predict_probability(feature_matrix):
            if propability > self.cut_off:
                classifications.append(1)
            else:
                classifications.append(0)
        return classifications


