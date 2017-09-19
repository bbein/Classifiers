"""
Core functions for Classifiers that can be shared between multiple classifiers.
"""

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

    """
    #ToDo: Add error if len(marix[0]) != len(vector)
    #ToDo: Add error if len(marix[0]) != Len(marix[x])

    #ToDo: Add example in this comment
    return [sum([column[x]*vector[x] for x in range(len(vector))]) for column in matrix]
