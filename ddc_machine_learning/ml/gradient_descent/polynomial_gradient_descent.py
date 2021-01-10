import random
import numpy as np
import matplotlib.pyplot as plt

class Polynomial_GD():
    """
    Class to predict a polynomial function from data.
    Data should be floats.
    Uses packages numpy, random and matplotlib.pyplot.
    
    ::param n: (int) Number of coefficients in predicted polynomial
        The maximum power of the predicted polynomial is n - 1
    ::param learning_rate: (float) Learning rate of the gradient descent, default = 0.0001
    ::param early_stop: (float) Stops if loss difference of 2 steps < early_stop, default = 1e-04
    ::param steps: (int) maximum number of steps of gradient descent, default = 100000
    """
    
    def __init__(
        self,
        n = 2,
        learning_rate = 0.0001,
        early_stop = 1e-4,
        steps = 100000,
    ):
        """
        Initialisation function for predicting a polynomial.
        
        ::param n: (int) Number of coefficients in predicted polynomial
            The maximum power of the predicted polynomial is n - 1
        ::param learning_rate: (float) Learning rate of the gradient descent, default = 0.0001
        ::param early_stop: (float) Stops if loss difference of 2 steps < early_stop, default = 1e-04
        ::param steps: (int) maximum number of steps of gradient descent, default = 100000
        """
        self.n = n
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.steps = steps
        self.coefficients = self.random_coefficients(n)
        self.loss = np.array([])
        self.x_values = np.array([])
        self.y_values = np.array([])
        self.old_loss = 0

        
    def random_coefficients(self, n=3, max_range = 10):
        """
        Randomly creates polynomial coefficients, of size n.
        Coefficents are random uniform ranging from (-1)*max_range, max_range
            Need to find a way to randomise the coefficients better.


        ::param n: (int) Size of number of coefficients.
                            Max exponent power is n-1
        ::param max_range: (float)
        ::returns: (list[floats])
        """
        return np.random.uniform(-1*max_range, max_range, n) 

    
    def f(self, x, coeffs, jitter = 0):
        """
        Function to evaluate a polynomial function.
            Can also add jitter and noise.

        ::param x: (float) 
        ::param coeffs: (np.array), coeffs of polynomial, where the index correspond to teh power
        ::param jitter: (float), uniform half range of noise, default = 0
        ::return: (float)
        """
        return np.polyval(np.flip(coeffs), x) + random.uniform(-jitter,jitter)


    def simulate_x_values(self, minimum = -10, maximum = 10, length = 100):
        """
        Function to create a list of random numbers.

        ::param minimum: (float), default -10
        ::param maximum: (float), default 10
        ::param lenth: (int), length of returned list, default 100
        ::return: (list[float]), sorted in ascending order
        """
        return np.sort(np.random.uniform(minimum, maximum, length) )


    def loss_mse(self, coeffs, x_values, y_values):
        """
        Loss function of a polynomial.

        ::param coeffs: (list[float]) 
        ::param x_values: (list[float]) 
        ::param y_values: (list[float]) 
        ::return: (float)    
        """
        return np.mean(pow(self.f(x_values, coeffs) - y_values, 2))

    def gradient_calculation(self, coefficients, x_values, y_values):
        """
        Function to return the gradient of a polynomial MSE loss.

        ::param coefficients: (list[floats])
        ::param x_values: (list[floats])
        ::param y_values: (list[floats])    
        """
        gradient_coeffs =  np.array([0]*len(coefficients))

        for xi in range(len(x_values)):
            x = x_values[xi]
            power_array = np.power(
                np.array([x]*len(coefficients)), np.array(range(len(coefficients))))

            diff = (2/len(x_values))*(self.f(x, coefficients) - y_values[xi])
            gradient_coeffs = gradient_coeffs + np.multiply(diff, power_array)

        return gradient_coeffs


    def gradient_descent(
        self,
        coeffs, 
        x_values, y_values):
        """
        Function to predict a polynomial to fit given x and y values.

        ::param coeffs: (numpy array) position of the array corresponds to the exponent power.
        ::param x_values: (numpy array) 
        ::param y_values: (numpy array) 
        ::param steps: (int) number of gradient calculations, and updates to the coefficients, default = 100000
        ::param learning_rate: (float) weight applied to the gradient, default = 0.0001
        ::param cut_off: (float) when, for step n and n+1, mse(n) - mse(n-1) <= cut_off 
        """
        old_loss = self.old_loss
        mse = self.loss

        for i in range(self.steps):
            new_loss = self.loss_mse(coeffs, x_values, y_values)
            mse = np.append(mse, new_loss)
            if abs(new_loss - old_loss) <= self.early_stop:
                print(f"Early cut off, difference of losses between steps is less that {self.early_stop}.")
                break
            old_loss = new_loss

            coeffs = coeffs - (self.learning_rate)*self.gradient_calculation(coeffs, x_values, y_values)

        mse = np.append(mse, self.loss_mse(coeffs, x_values, y_values))
        self.coefficients = coeffs
        self.loss = mse


    def fit(self, X, y):
        """
        Fit the data into a polynomial.
        
        """
        self.x_values = X
        self.y_values = y
        self.gradient_descent(self.coefficients, X, y)

        
    def predict(self, X):
        """
        Fit the data into a polynomial.
        
        """        
        return self.f(X, self.coefficients)


    def plot_loss(self):
        """
        Function to plot the loss of a gradient descent process.
        """
        plt.plot(self.loss[10:], 'g+', label = "loss")
        plt.plot(self.loss[10:], 'r--', label = "loss (smooth)")
        plt.title(f"Graph of loss after {len(self.loss)} steps of Gradient Descent.")
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


    def plot_polynomial(self):
        """
        Function to plot the variables of 2 lists.

        ::param x_values: (list[int])
        ::param y_values: (list[int])
        """
        plt.scatter(self.x_values, self.y_values)
        plt.title(f"Graph of polynomial between {np.floor(min(self.x_values))} and {np.ceil(max(self.x_values))}")
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.show()


    def plot_actual_predicted(self):
        """
        Function to plot actual values and predicted values.

        ::param coeffients: (list[floats])
        ::param x_values: (list[floats])
        ::param y_values: (list[floats])    
        """
        predicted = [self.f(x, self.coefficients) for x in self.x_values]

        plt.scatter(self.x_values, self.y_values, label = "Actual data", c = 'b')
        plt.plot(self.x_values, predicted, label = "Predicted data", c =  'r')
        plt.title(f"Graph of Prediected and Actual data points.")
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.legend()
        plt.show()