from numpy.linalg import eig as numpy_eigenvalues


def size(matrix):
    """
    Return the size/dimensions of a matrix.
    To run 
        --size(matrix)

    ::param matrix: (list[list])

    ::returns: (tuple) first element number of rows, second number of columns.
    """
    assert len(set([len(row) for row in matrix])) == 1, \
        """Error: Not all matrix rows have the same length."""

    rows = len(matrix)
    columns = len(matrix[0])

    return (rows, columns)


def transpose(matrix):
    """
    Return the Transpose of a matrix.
    To run 
        --transpose(matrix)

    ::param matrix: (list[list])

    ::returns: (list[list])
    """
    rows, columns = size(matrix)

    matrix_new = []
    for column in range(columns):
        matrix_new.append([row[column] for row in matrix])

    assert (columns, rows) == size(matrix_new), \
        """Error in code, the transposed matrix is of the wrong dimensions."""
    return matrix_new


def split_into_columns(matrix):
    """
    Function to split Matrix into columns.
    Instead of row -matrix, return column -matrix.
    Similiar to getting the transpose of a matrix.
    
    ::param matrix: (list[list])
    
    ::return: (list[list])
    """
    return transpose(matrix)


def mean(list_):
    """
    Return the mean of a list.

    ::param list_: (list)

    ::returns: (numeric)
    """
    return sum(list_)/len(list_)


def dot_product(col1, col2):
    """
    Return the Covariance of between two columns.

    ::param col1: (list)
    ::param col2: (list)

    ::returns: (numeric)
    """
    assert len(col1) == len(col2),\
        "Error: Columns should be the same size"
        
    list_new = []
    for i in range(len(col1)):
        list_new.append(col1[i]*col2[i])

    return sum(list_new)


def covariance_cols(col1,col2):
    """
    Return the Covariance of between two columns.

    ::param col1: (list)
    ::param col2: (list)

    ::returns: (numeric)
    """
    assert len(col1) == len(col2),\
        "Error: Columns should be the same size"

    mean1 = mean(col1)
    mean2 = mean(col2)
    
    # Centre the data around 0
    col1 = [val-mean1 for val in col1]
    col2 = [val-mean2 for val in col2]
    
    return dot_product(col1, col2)/len(col1)


def scaler_product(matrix, value):
    """
    Return a matrices*value.
    For a (m,n)-dimensional matrix, returns a (n,n)-dimensional.
    To run 
        --scaler_product(matrix, value)
        
    ::param matrix: (list[list])
    ::param value: (numeric)

    ::returns: (list[list])
    """
    matrix_new = []
    for row in matrix:
        matrix_new.append([value*element for element in row])
    return matrix_new


def multiply(matrix, matrix_other):
    """
    Return the product of two matrices.
    For a (m,n)-dimensional matrix, returns a (n,n)-dimensional.
    To run 
        --covariance(matrix)
        
    ::param matrix: (list[list])
    ::param matrix_other: (list[list])

    ::returns: (list[list])
    """

    matrix_size = size(matrix)
    matrix_other_size = size(matrix_other)
    assert matrix_size[1] == matrix_other_size[0], \
        """Error: Cannot multiply the two matrices together."""

    matrix_new = []
    for row in range(matrix_size[0]):
        row_new = []
        for column in range(matrix_other_size[1]):
            row_new.append(
                sum([a * b for a, b in zip(matrix[row], [matrix_[column] for matrix_ in matrix_other])]))
        matrix_new.append(row_new)

    return matrix_new


def remove_mean(matrix):
    """
    Return a matrix whose mean of each column is 0.
    For a (m,n)-dimensional matrix, returns a (m,n)-dimensional matrix.
    To run 
        --remove_mean(matrix)
        
    ::param matrix: (list[list])

    ::returns: (list[list])
    """
    matrix_transpose = transpose(matrix)
    matrix_new = []
    means = []
    for i in range(len(matrix_transpose)):
        means.append(mean(matrix_transpose[i]))
        matrix_new.append([col - means[i] for col in matrix_transpose[i]])
    return transpose(matrix_new), means


def covariance(matrix):
    """
    Return the Covariance of a matrix.
    For a (m,n)-dimensional matrix, returns a (n,n)-dimensional.
    To run 
        --covariance(matrix)
        
    ::param matrix: (list[list])

    ::returns: (list[list])
    """
    return scaler_product(multiply(transpose(matrix), matrix), (1/len(matrix)))


def order_eigenvalues(cov, eigenvectors):
    """
    Return a reordered version of the eigenvectors, with the eigenvectors returning
    in the order of highest eigenvalues to lowest.
    Issue:
        If two eigenvalues are the same...
    
    ::param cov: (list[list])
    ::param eigenvectors: (list of eigenvectors)

    ::returns: (list[list])
    """
    
    eigendict = {}
    for col in eigenvectors:
        eigenvalue = multiply(cov, transpose([col]))[0]/col[0]
        eigendict[eigenvalue[0]] = col
    
    eigenvalues = list(eigendict.keys())
    eigenvalues.sort(reverse = True)
    
    return list([eigendict[eigenvalue] for eigenvalue in eigenvalues]), eigenvalues


def get_eigenvalues(matrix):
    """
    Return the eigenvalues of a matrix.
        
    ::param matrix: (list[list])

    ::returns: (list[list])
    """
    _, eigenvectors = numpy_eigenvalues(matrix)

    return transpose(list([list(eig) for eig in eigenvectors]))