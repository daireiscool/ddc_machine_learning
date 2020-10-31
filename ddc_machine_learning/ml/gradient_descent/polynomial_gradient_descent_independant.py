import random

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
        initial_coefficients = []
    ):
        """
        Initialisation function for predicting a polynomial.
        
        ::param n: (int) Number of coefficients in predicted polynomial
            The maximum power of the predicted polynomial is n - 1
        ::param learning_rate: (float) Learning rate of the gradient descent, default = 0.0001
        ::param early_stop: (float) Stops if loss difference of 2 steps < early_stop, default = 1e-04
        ::param steps: (int) maximum number of steps of gradient descent, default = 100000
        ::param initial_coefficients: (list) Initial coefficients, len(initial_coefficients) == n.
            If null, will start with all oefficients of 1.
        """
        self.n = n
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.steps = steps
        self.coefficients = initial_coefficients if len(initial_coefficients) == n else [1]*n
        self.loss = []
        self.x_values = []
        self.y_values = []
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
        coeff = []
        for i in range(n):
            coeff += [random.uniform(-max_range,max_range)]
        return coeff


    def f(self, x, coeffs, jitter = 0):
        """
        Function to simulate a polynomial function.

        ::param x: (float) 
        ::param jitter: (float), uniform half range of noise, default = 10
        ::param coeffs: (np.array), coeffs of polynomial, ,default = [1,-4,2]
        ::return: (float)
        """
        y = 0
        for i in range(len(coeffs)):
            y += coeffs[i]*(x**i)

        return y + random.uniform(-jitter,jitter)


    def simulate_x_values(self, minimum = -10, maximum = 10, length = 100):
        """
        Function to create a list of random numbers.

        ::param minimum: (float), default -10
        ::param maximum: (float), default 10
        ::param lenth: (int), length of returned list, default 100
        ::return: (list[float])
        """
        val = []
        for i in range(length):
            val += [random.uniform(minimum,maximum)]
        val.sort()
        return val

    
    def loss_mse(self, coeffs, x_values, y_values):
        """
        Loss function of a polynomial.

        ::param coeffs: (list[float]) 
        ::param x_values: (list[float]) 
        ::param y_values: (list[float]) 
        ::return: (float)    
        """
        def f_mse(x, y):
            return (self.f(x, coeffs) - y)**2

        loss = [f_mse(x_values[xi], y_values[xi]) for xi in range(len(x_values))]
        return sum(loss)*(1/len(x_values))


    def gradient_calculation(self, coefficients, x_values, y_values):
        """
        Function to return the gradient of a polynomial MSE loss.

        ::param coefficients: (list[floats])
        ::param x_values: (list[floats])
        ::param y_values: (list[floats])    
        """
        gradient_coeffs =  [0]*len(coefficients)

        for xi in range(len(x_values)):
            x = x_values[xi]
            diff = (2/len(x_values))*(self.f(x, coefficients) - y_values[xi]) 
            for coeff_loc in range(len(coefficients)):
                gradient_coeffs[coeff_loc] += diff*((x_values[xi])**coeff_loc)

        return gradient_coeffs


    def gradient_descent(
        self,
        coeffs, 
        x_values, y_values):
        """
        Function to predict a polynomial to fit given x and y values.

        ::param coeffs: (list[floats]) position of the array corresponds to the exponent power.
        ::param x_values: (list[floats]) 
        ::param y_values: (list[floats]) 
        """
        old_loss = self.old_loss
        mse = self.loss

        for i in range(self.steps):
            new_loss = self.loss_mse(coeffs, x_values, y_values)
            mse += [new_loss]
            if abs(new_loss - old_loss) <= self.early_stop:
                print(f"Early cut off, difference of losses between steps is less that {self.early_stop}.")
                break
            old_loss = new_loss

            gradient = [self.learning_rate*coeff for coeff in self.gradient_calculation(coeffs, x_values, y_values)]
            
            assert len(gradient) == len(coeffs), \
                "Gradient adn coefficients have different lengths."
            
            for i in range(len(coeffs)):
                coeffs[i] = coeffs[i] - gradient[i]

        mse += [self.loss_mse(coeffs, x_values, y_values)]
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
        return [self.f(x, self.coefficients) for x in X]