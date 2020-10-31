# Gradient Descent for Polynomial Data


### Notes:
* The polynomials are in the form of a numpy array, where the polynomial c+bx+ax^2 = y will be rewritten can be rewritten as np.array([c,b,a]), where the position of the array corresponds to the power of the x-value.  
Ive created two versions, one with numpy, and another that just uses list comprehensions.

##### To improve:
* Different learning rates, eg degrading
* Try remove all for loops
* Currently only works on 1-dimensional data
