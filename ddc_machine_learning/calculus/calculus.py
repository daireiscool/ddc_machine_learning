"""
Inputted polynomials are dictionaries.
"""

def differentiate(polynomial):
    """
    Function to differentiate a polynomial.
    The polynomial is in the form a python dict.
    The key of the dict corresponding to a exponent, and the value the corresponding coefficient.
        
    ::param polynomial: (dict) key: float, value: float
    ::return: (dict)
    """
    assert isinstance(polynomial, dict), \
        "Error: The input poymomial was not a python dictionary"
    assert all(isinstance(x, (float, int)) for x in polynomial.keys()), \
        "Error: One of the exponents is not numeric"
    assert all(isinstance(x, (float, int, complex)) for x in polynomial.values()), \
        "Error: One of the coefficients is not numeric"

    return dict([(a-1, b*a) for (a, b) in polynomial.items() if a != 0])


def evaluate(polynomial, value):
    """
    Function where, given an polynomial (in the form of a python dict), and a value,
    will evaluate the polynomial in terms of the value.
    # For large exponentials may get a OverflowError.
    
    ::param polynomial: (dict) polynomial in dict form
    ::param value: (float) Must be a non-NaN float
    ::return: (float)
    """
    assert isinstance(polynomial, dict), \
        "Error: The input poymomial was not in the form of a python dictionary"
    assert all(isinstance(x, (float, int)) for x in polynomial.keys()), \
        "Error: One of the exponents is not numeric"
    assert all(isinstance(x, (float, int, complex)) for x in polynomial.values()), \
        "Error: One of the coefficients is not numeric"
    assert isinstance(value, (float, int, complex)), \
        "Error: The input value was not numeric"

    return sum([b*((value)**a) for (a, b) in polynomial.items()])


