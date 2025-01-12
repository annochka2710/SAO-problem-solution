from draw import *
from functions import *
from sao import *
import numpy as np


def main():
    SearchAgents_no = 100
    Max_iteration = 100 
    dim = 2

    arr = []
    arr.append(("Himmelblau function:",    "Himmelblau",   fH      , -3, 4))
    arr.append(("Rosenbrock function:",    "Rosenbrock",   fR      , -2, 2))
    arr.append(("Sphere Function:",        "Sphere",       spheref , -1, 1))
    arr.append(("MICHALEWICZ FUNCTION:",   "MICHALEWICZ",  michal  ,  0, 4))
    arr.append(("ZAKHAROV FUNCTION:",      "ZAKHAROV",     zakharov, -5, 10))
    arr.append(("GRIEWANK FUNCTION:",      "GRIEWANK",     griewank, -50, 50))

    for cort in arr :
        np.random.seed(0) #for testing
        [name, flag, func, lb, ub] = cort
        print(name)
        [xmin, f, neval, coords] = SAO(SearchAgents_no, Max_iteration, lb, ub, dim, func) #функция Химмельблау
        print('x_min = (', xmin[0,0], ',', xmin[1,0],'),','f_min = ', f, ', neval = ', neval)
        draw(coords, xmin, len(coords), func, flag, lb, ub)


if __name__ == '__main__':
    main()