"""
Core functions for Classifiers that can be shared between multiple classifiers.
"""

def dot_product_vector_vector(vector_1, vector_2):
    """
    Dot product between two vectors.
    Both vectors need to be of the same dimention

    # Parameters

    vector_1 : list-like, with a list of all vector data points
    vector_2 : list-like, with a list of all vector data points

    # Returns

    float, of the dot product

    # Example:

    >>> vector_1 = [2, 3]
    >>> vector_2 = [1, 2]
    >>> dot_product_vector_vector(vector_1, vector_2)
    8
    """
    if len(vector_2) != len(vector_1):
        message = "The length of the vectors has to be the same!"
        message += "Length vector 1: {0}. ".format(len(vector_1))
        message += "Length vector 2: {0}. ".format(len(vector_2))
        raise ValueError(message)
    return sum(x*y for x, y in zip(vector_1, vector_2))

def dot_product_matrix_vector(matrix, vector):
    """
    Column vise dot product between a matrix and vector.
    The matrix needs as many columns as the vectore has data points.
    The matrix can have an unlimited number of rows

    # Parameters

    matrix : list-like, with the matrix data e.g [[x_11, x_12, ...], [x_21, x_22, ...], ...]
    vector : list-like, with a list of all vector data points

    # Returns

    List of the column vise dot product

    # Example:

    >>> matrix = [[1, 2], [2, 3], [3, 4]]
    >>> vector = [1, 2]
    >>> dot_product_matrix_vector(matrix, vector)
    [5, 8, 11]
    """
    # Check that the length of the vector and matrix elements match
    if len(matrix[0]) != len(vector):
        message = "The length of the vector has to be the same as the length of each matrix row!"
        message += "Length vector: {0}. ".format(len(vector))
        message += "Length matrix row 0: {0}".format(len(matrix[0]))
        raise ValueError(message)
    for i, row in enumerate(matrix):
        if len(matrix[0]) != len(row):
            message = "The length of each row in the matrix has to be the same!"
            message += "Length matrix row {0}: {1}. ".format(i, len(row))
            message += "Length matrix row 0: {0}".format(len(matrix[0]))
            raise ValueError(message)

    return [dot_product_vector_vector(column, vector) for column in matrix]

