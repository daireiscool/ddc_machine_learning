from .matrix_functions import *

class PCA():
    """
    Class for PCA operations.
    The matrix is represented as a nested list, where each sublist is a row.
    
    ::param n_components: (int) Number of Principle Components to return
    """
    def __init__(self, n_components):
        """
        Initialisation function for the PCA Class.
        The default model has no data in it.
        
        ::param n_components: (int) Number of Principle Components to return
        
        ::returns: (Class PCA)
        """
        self.n_components = n_components
        self.eigenvalues = []
        self.eigenvectors = []
        self.mean = []
        

    def variance_explained(self):
        """
        Print the variance explained by the eigenvectors. 

        ::param eigenvalues: (list)
        """

        explained_variances = []
        for i in range(len(values)):
            explained_variances.append(self.eigenvalues[i] / np.sum(values))

        variance_explain = 0
        for i in range(self.n_components):
            variance = round(explained_variances[i]*100, 4)
            variance_explained += variance
            print(f"PC{i} explains {variance}% of the variance.")
        print(f"The top {n_components} Principle Components explain {variance_explain}% of the variance.")


    def fit(self, matrix):
        """
        Fit the matrix onto PCA.
            
        ::param matrix: (list[list])
        """
        matrix, means = remove_mean(matrix)
        cov = covariance(matrix)
        eigenvectors = get_eigenvalues(cov)
        eigenvectors, eigenvalues = order_eigenvalues(cov, eigenvectors)

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.means = means


    def transform(self, matrix):
        """
        Transform the matrix, based on the pre-trained fitted model.
            
        ::param matrix: (list[list])
        """
        _, columns = size(matrix)
        
        assert self.n_components < columns, \
            """
            The number of Principle Components wanted is more than the number of columns.
            Please change n_components of the class.
            """

        matrix_pca = []
        for i in range(self.n_components):
            matrix_pca.append([col[0] for col in multiply(matrix, transpose([self.eigenvectors[i]]))])
        return transpose(matrix_pca)



    def reverse(self, matrix):
        """
        Reverse the PCA process.
        Data information will be lost.
            
        ::param matrix: (list[list])

        ::returns: (list[list])
        """

        return multiply(matrix, self.eigenvectors[:self.n_components])
