import numpy as np
import math

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v

# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v

def spheref(xx):
    return sum(xi ** 2 for xi in xx)

# MICHALEWICZ FUNCTION
def michal(xx, m=10):
    d = len(xx)
    total = 0
    for ii in range(1, d + 1):  # MATLAB indices start at 1
        xi = xx[ii - 1]         # Adjust for 0-based indexing in Python
        term = math.sin(xi) * (math.sin(ii * xi ** 2 / math.pi)) ** (2 * m)
        total += term
    return -total

# ZAKHAROV FUNCTION
def zakharov(xx):
    d = len(xx)
    sum1 = sum(xi ** 2 for xi in xx)
    sum2 = sum(0.5 * (i + 1) * xx[i] for i in range(d))
    return sum1 + sum2 ** 2 + sum2 ** 4

# GRIEWANK FUNCTION
def griewank(xx):
    d = len(xx)
    sum_term = sum(xi ** 2 / 4000 for xi in xx)
    prod_term = math.prod(math.cos(xx[i] / math.sqrt(i + 1)) for i in range(d))
    return sum_term - prod_term + 1
