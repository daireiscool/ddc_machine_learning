import csv


class Matrix():
    """
    Class for matrix operations.
    The matrix is represented as a nested list, where each sublist is a row.
    
    ::param matrix: (list) nested list
    """
    def __init__(self, matrix = [[]]):
        """
        Initialisation function for the Matrix Class.
        Creates a Matrix, and checks that the size is correct.
        The default matrix is empty.
        
        ::param matrix: (list) List of lists, default = [[]]
        
        ::returns: (Class Matrix)
        """
        self.matrix = matrix
        self.size()

    def __add__(self, OtherMatrix):
        """
        Add two Matrices together.
        To use either run 
            --Matrix.__add__(OtherMatrix)
            --Matrix + OtherMatrix.

        ::param OtherMatrix: (Class Matrix)

        ::return: (Class Matrix)
        """

        assert self.size() == OtherMatrix.size(), \
            """Error: The two matrices are of different dimensions."""
        matric_new = []
        matrix_other = OtherMatrix.show()

        for row in range(len(self.matrix)):
            matric_new.append([sum(x) for x in zip(self.matrix[row], matrix_other[row])])
        return Matrix(matric_new)

    def __sub__(self, OtherMatrix):
        """
        Sub two Matrices together.
        To use either run 
            --Matrix.__sub__(OtherMatrix)
            --Matrix - OtherMatrix.

        ::param OtherMatrix: (Class Matrix)

        ::return: (Class Matrix)
        """
        assert self.size() == OtherMatrix.size(), \
            """Error: The two matrices are of different dimensions."""

        MatrixNegative = OtherMatrix.scaler_product(-1)
        return self.__add__(MatrixNegative)

    def __mul__(self, OtherMatrix):
        """
        Multiply two Matrices together.
        Must have the correct dimensions.
        Matrix has dimensions of (x, y)
        OtherMatrix has dimensions of (y, z)
        To use either run 
            --Matrix.__mul__(OtherMatrix)
            --Matrix * OtherMatrix.

        ::param OtherMatrix: (Class Matrix)

        ::return: (Class Matrix)
        """
        matrix_size = self.size()
        matrix_other_size = OtherMatrix.size()
        assert matrix_size[1] == matrix_other_size[0], \
            """Error: Cannot multiply the two matrices together."""

        matrix_other = OtherMatrix.show()
        matrix_new = []
        for row in range(matrix_size[0]):
            row_new = []
            for column in range(matrix_other_size[1]):
                row_new.append(
                    sum([a * b for a, b in zip(self.matrix[row], [matrix[column] for matrix in matrix_other])]))
            matrix_new.append(row_new)

        return Matrix(matrix_new)    

    def __eq__(self, OtherMatrix):
        """
        Are two Matrices equal.
        To use either run 
            --Matrix.__eq__(OtherMatrix)
            --Matrix == OtherMatrix.

        ::param OtherMatrix: (Class Matrix)

        ::return: (Class Matrix)
        """
        return self.matrix == OtherMatrix.show()

    def __pow__(self, power):
        """
        Get the power of a Matrix.
        To use either run 
            --Matrix.__pow__(power)
            --Matrix**power.

        ::param power: (int)

        ::return: (Class Matrix)
        """
        assert type(power) == int, \
            "Error: Power must be an integer"
        row, column = self.size()

        assert row == column, \
            "Error: Can only get the power of a square matrix."

        if power == -1:
            return self.inverse_2x2()
        elif power == 0:
            return self * self.inverse_2x2()
        elif power > 1:
            M = self
            for i in range(power-1):
                M = M * self
            return M
        else:
            raise Exception(f"Error: Cannot get the power of {power}")
    
    def show(self):
        """
        Return the matrix in the form of a nested list, where each sublist is a row.
        To run 
            --Matrix.show()

        ::return: (list) List of lists
        """
        return self.matrix
    
    def size(self):
        """
        Return the size/dimensions of a matrix.
        To run 
            --Matrix.size()

        ::returns: (tuple) first element number of rows, second number of columns.
        """
        assert len(set([len(row) for row in self.matrix])) == 1, \
            """Error: Not all matrix rows have the same length."""

        rows = len(self.matrix)
        columns = len(self.matrix[0])

        return (rows, columns)

    def scaler_product(self, value):
        """
        Multiply a Matrix by a scaler value.
        To run 
            --Matrix.scaler_product(value)
            
        ::param value: (numeric)

        ::returns: (Class Matrix) value * Matrix
        """
        matrix_new = []
        for row in self.matrix:
            matrix_new.append([value*element for element in row])
        return Matrix(matrix_new)

    def determinant_2x2(self):
        """
        Return the determinant of a (2,2)-matrix.
        To run 
            --Matrix.determinant_2x2()

        ::returns: (numeric)
        """
        assert self.size() == (2,2), \
            """Error: The inputted Matrix is not (2,2)-dimensional."""

        return self.matrix[0][0]*self.matrix[1][1] - self.matrix[0][1]*self.matrix[1][0]

    def inverse_2x2(self):
        """
        Return the inverse of a (2,2)-matrix.
        To run 
            --Matrix.inverse_2x2()

        ::returns: (Class Matrix)
        """
        determinant = self.determinant_2x2()

        assert determinant != 0, \
            """Error: The matrix is not invertable."""

        MatrixNew = Matrix([
            [self.matrix[1][1], -1*self.matrix[0][1]],
            [-1*self.matrix[1][0], self.matrix[0][0]]])
        return MatrixNew.scaler_product(1/determinant)
    
    def transpose(self):
        """
        Return the Transpose of a matrix.
        To run 
            --Matrix.transpose()

        ::returns: (Class Matrix)
        """
        rows, columns = self.size()

        matrix_new = []
        for column in range(columns):
            matrix_new.append([row[column] for row in self.matrix])
        MatrixNew = Matrix(matrix_new)

        assert (columns, rows) == MatrixNew.size(), \
            """Error in code, the transposed matrix is of the wrong dimensions."""
        return MatrixNew

    def save(self, location):
        """
        Saves a Matrix to storage as csv.
        To run 
            --Matrix.save(location)
            
        ::param location: (string) path and location of file to save

        """
        with open(location, "w", newline='') as file:
            writer = csv.writer(file)
            for row in self.matrix:
                writer.writerow(row)
            file.close()
        print(f"Saved to {location}.")
            
    def load(self, location):
        """
        Load a Matrix from a csv file.
        Replaces the matrix with the loaded data.
        To run create an empty Matrix, and load the data to overwrite the empty Matrix.
        To run 
            M = Matrix()
            M.load("test.csv")
            
        ::param location: (string) path and location of file to load

        """
        with open(location, "r", newline='') as file:
            matrix_new = list(csv.reader(file, quoting=csv.QUOTE_NONNUMERIC))
            file.close()
        print(f"Loaded matrix with dimension: {self.size()}")
        self.matrix = matrix_new