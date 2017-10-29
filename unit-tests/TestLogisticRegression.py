"""
Unit-Test module for the Logistic Regression Classifier.
"""
import unittest
import math

from Classifiers.LogisticRegressionClassifier import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    """
    Testing the predict_probability function
    """
    def __init__(self, *args, **kwargs):
        """
        Testing the predict_probability function and select the test data set
        """
        super(TestLogisticRegression, self).__init__(*args, **kwargs)
        self.model = LogisticRegression()

    def test_predict_probability_result(self):
        """
        Testing the predict_probability function results are correct
        """
        self.model._coefficients = [1., 3., -1.]
        feature_matrix = [[1., 2., 3.], [1., -1., -1]]
        correct_results = [1. / (1 + math.exp(-4)), 1. / (1 + math.exp(1))]
        results = self.model.predict_probability(feature_matrix)
        self.assertEqual(correct_results, results)

    def test_predict_result(self):
        """
        Testing the predict function results are correct
        """
        self.model._coefficients = [1., 3., -1.]
        feature_matrix = [[1., 2., 3.], [1., -1., -1]]
        correct_results = [1, 0]
        results = self.model.predict(feature_matrix)
        self.assertEqual(correct_results, results)

    def test_predict_result_edge(self):
        """
        Testing the predict function returns 0 if propability is equals `cut_off`
        """
        self.model._coefficients = [1., 3., -1.]
        feature_matrix = [[0., 0., 0.]]
        correct_results = [0]
        results = self.model.predict(feature_matrix)
        self.assertEqual(correct_results, results)

    def test_train_result(self):
        """
        Testing the training function works correctly
        """
        feature_matrix = [[0., -1., 3.], [0., 0., 1.], [1., 0., 0.5]]
        targets = [0, 0, 1]
        self.model.train(feature_matrix, targets)
        self.assertAlmostEqual(self.model.predict(feature_matrix), targets)

    def test_train_target_features(self):
        """
        Testing the training function raises an error if there are less targets than deature rows
        """
        feature_matrix = [[0., -1., 3.], [0., 0., 1.], [1., 0., 0.5]]
        targets = [0, 0, 1, 0]
        self.assertRaises(ValueError, self.model.train, feature_matrix, targets)

    def test_train_features(self):
        """
        Testing the training function raises an error if not every row has every feature in
        the feature matrix
        """
        feature_matrix = [[0., -1., 3.], [0., 0., 1.], [1., 0.]]
        targets = [0, 0, 1]
        self.assertRaises(ValueError, self.model.train, feature_matrix, targets)

    def test_train_targets(self):
        """
        Testing the training function raises an error if any target is not 0 or 1
        """
        feature_matrix = [[0., -1., 3.], [0., 0., 1.], [1., 0., 0.5]]
        targets = [0, 0, 2]
        self.assertRaises(ValueError, self.model.train, feature_matrix, targets)

if __name__ == '__main__':
    unittest.main()
