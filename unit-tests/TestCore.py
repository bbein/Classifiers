"""
Unit-Test module for the Classifier Core functions.
"""
import unittest

from Classifiers import Core

class TestDotProductVectorVector(unittest.TestCase):
    """
    Testing the dot product function for two vectores
    """

    def test_result(self):
        """
        Testing the dot product function for matrices and vectores results
        """
        vector_1 = [2, 3]
        vector_2 = [1, 2]
        result = 8
        self.assertEqual(Core.dot_product_vector_vector(vector_1, vector_2), result)

    def test_vector1_unequal_vector2(self):
        """
        Testing the dot product function for two vectores raises ValueError
        if len(vector_1) != len(vector_2)
        """
        vector_1 = [1, 2, 3]
        vector_2 = [1, 2]
        self.assertRaises(ValueError, Core.dot_product_vector_vector, vector_1, vector_2)

class TestDotProductMatrixVector(unittest.TestCase):
    """
    Testing the dot product function for matrices and vectores
    """

    def test_result_1(self):
        """
        Testing the dot product function for matrices and vectores results 1
        """
        matrix = [[1, 2], [2, 3], [3, 4]]
        vector = [1, 1]
        result = [3, 5, 7]
        self.assertEqual(Core.dot_product_matrix_vector(matrix, vector), result)

    def test_result_2(self):
        """
        Testing the dot product function for matrices and vectores results 2
        """
        matrix = [[1, 2], [2, 3], [3, 4]]
        vector = [2, 1]
        result = [4, 7, 10]
        self.assertEqual(Core.dot_product_matrix_vector(matrix, vector), result)

    def test_result_3(self):
        """
        Testing the dot product function for matrices and vectores results 3
        """
        matrix = [[1, 2], [2, 3], [3, 4]]
        vector = [1, 2]
        result = [5, 8, 11]
        self.assertEqual(Core.dot_product_matrix_vector(matrix, vector), result)

    def test_matrix_unequal_vector(self):
        """
        Testing the dot product function for matrices and vectores raises ValueError
        if len(vector) != len(matrix[0])
        """
        matrix = [[1, 2], [2, 3], [3, 4]]
        vector = [1]
        self.assertRaises(ValueError, Core.dot_product_matrix_vector, matrix, vector)

    def test_matrix_unequal_matrix(self):
        """
        Testing the dot product function for matrices and vectores raises ValueError
        if len(matrix[i]) != len(matrix[0])
        """
        matrix = [[1, 2, 3], [2, 3], [3, 4]]
        vector = [1, 2]
        self.assertRaises(ValueError, Core.dot_product_matrix_vector, matrix, vector)

if __name__ == '__main__':
    unittest.main()
