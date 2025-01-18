from draw import *
from functions import *
from sao import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os


def getFunctions():
    arr = []
    # Кортеж: (Имя функции, приставка файла, функция, нижняя граница, верхняя ганица, значения минимума
    arr.append(("Himmelblau function:",    "Himmelblau",   fH      , -3, 4   , [(3, 2), (-2.805118, 3.283186), (-3.779310, -3.283186), (3.584428, -1.848126)]))
    arr.append(("Rosenbrock function:",    "Rosenbrock",   fR      , -2, 2   , [(1, 1)]))
    arr.append(("Sphere Function:",        "Sphere",       spheref , -1, 1   , [(0, 0)]))
    arr.append(("MICHALEWICZ FUNCTION:",   "MICHALEWICZ",  michal  ,  0, 4   , [(2.20, 1.57)]))
    arr.append(("ZAKHAROV FUNCTION:",      "ZAKHAROV",     zakharov, -5, 10  , [(0, 0)]))
    arr.append(("GRIEWANK FUNCTION:",      "GRIEWANK",     griewank, -50, 50 , [(0, 0)]))
    return arr

def distToMin(min, mins) :
    minDist = math.dist(mins[0], min)
    closestMin = mins[0]
    for m in mins :
        dist = math.dist(m, min)
        if (dist < minDist) : 
            minDist = dist
            closestMin = m
    return (minDist, closestMin)

def fcalcIter(SearchAgents_no, cort):
    dim = 2
    x = []
    y = []
    z = []
    [name, flag, func, lb, ub, mins] = cort
    for Max_iteration in range(10, 100):
        [xmin, f, neval, coords] = SAO(SearchAgents_no, Max_iteration, lb, ub, dim, func)
        [minDist, closestMin] = distToMin(xmin, mins)
        x.append(SearchAgents_no)
        y.append(Max_iteration)
        z.append(abs(func(closestMin) - f))
    return x,y,z

def faccuracyByAgentsAndIterations():
    arr = getFunctions()

    for cort in arr :
        x = []
        y = []
        z = []
        [name, flag, func, lb, ub, mins] = cort

        with ProcessPoolExecutor(max_workers= 13) as executor:
            results = list(executor.map(calcIter, range(10, 100), [cort] * len(range(10, 100))))

        # Разбираем результаты
        for xx, yy, zz in results:
            x.extend(xx)
            y.extend(yy)
            z.extend(zz)
        print(name)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(x, y, z, c=z, cmap='inferno', linestyle='None')
        ax.set_xlabel("Количество агентов")
        ax.set_ylabel("Количество итераций")
        ax.set_zlabel("Погрешность")
        if not os.path.exists("Faccuracy"): os.mkdir('Faccuracy')
        name = "./Faccuracy/" + flag + ".png"
        fig.savefig(name, dpi=600)

def calcIter(SearchAgents_no, cort):
    dim = 2
    x = []
    y = []
    z = []
    [name, flag, func, lb, ub, mins] = cort
    for Max_iteration in range(10, 100):
        [xmin, f, neval, coords] = SAO(SearchAgents_no, Max_iteration, lb, ub, dim, func)
        [minDist, closestMin] = distToMin(xmin, mins)
        x.append(SearchAgents_no)
        y.append(Max_iteration)
        z.append(minDist)
    return x,y,z

def accuracyByAgentsAndIterations():
    arr = getFunctions()

    for cort in arr :
        x = []
        y = []
        z = []
        [name, flag, func, lb, ub, mins] = cort

        with ProcessPoolExecutor(max_workers= 13) as executor:
            results = list(executor.map(calcIter, range(10, 100), [cort] * len(range(10, 100))))

        # Разбираем результаты
        for xx, yy, zz in results:
            x.extend(xx)
            y.extend(yy)
            z.extend(zz)
        print(name)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(x, y, z, c=z, cmap='inferno', linestyle='None')
        ax.set_xlabel("Количество агентов")
        ax.set_ylabel("Количество итераций")
        ax.set_zlabel("Погрешность")
        if not os.path.exists("accuracy"): os.mkdir('accuracy')
        name = "./accuracy/" + flag + ".png"
        fig.savefig(name, dpi=600)

def calcIter(SearchAgents_no, cort):
    dim = 2
    x = []
    y1 = []
    y2 = []
    [name, flag, func, lb, ub, mins] = cort
    Max_iteration = SearchAgents_no
    [xmin, f, neval, coords] = SAO(SearchAgents_no, Max_iteration, lb, ub, dim, func)
    [minDist, closestMin] = distToMin(xmin, mins)
    x.append(SearchAgents_no)
    y1.append(minDist)
    y2.append(abs(func(closestMin) - f))
    return x,y1, y2

def accuracyByAgentsAndIterations2d():
    arr = getFunctions()
    dim = 2
    if not os.path.exists("2daccuracy"): os.mkdir('2daccuracy')
    for cort in arr :
        x = []
        y1 = []
        y2 = []
        [name, flag, func, lb, ub, mins] = cort
        for SearchAgents_no in range(10, 500) :     
            Max_iteration = SearchAgents_no
            [xmin, f, neval, coords] = SAO(SearchAgents_no, Max_iteration, lb, ub, dim, func)
            [minDist, closestMin] = distToMin(xmin, mins)
            x.append(SearchAgents_no)
            y1.append(minDist)
            y2.append(abs(func(closestMin) - f))
        print(name)

        plt.clf()
        plt.plot(x, y1, 'red', label ='Погрешность x')
        plt.xlabel("Количество агентов, итераций")
        plt.ylabel("Погрешность")
        plt.legend()
        name = "./2daccuracy/" + flag + "f.png"
        plt.savefig(name, dpi=600)

        plt.clf()
        plt.plot(x, y2, 'blue', label ='Погрешность f(x)')
        plt.xlabel("Количество агентов, итераций")
        plt.ylabel("Погрешность")
        plt.legend()
        name = "./2daccuracy/" + flag + "x.png"
        plt.savefig(name, dpi=600)

def main():
    SearchAgents_no = 100
    Max_iteration = 100 
    dim = 2
    arr = getFunctions()
    file = open('functions.csv','w')
    print('Имя функции; Истинный минимум; Расчитанный минимум; Значение функции; Шаги; Погрешность')
    file.write('Имя функции; Истинный минимум; Расчитанный минимум; Значение функции; Шаги; Погрешность\n')
    for cort in arr :
        np.random.seed(0)
        [name, flag, func, lb, ub, mins] = cort
        [xmin, f, neval, coords] = SAO(SearchAgents_no, Max_iteration, lb, ub, dim, func)
        [minDist, closestMin] = distToMin(xmin, mins)
        #  xmin
        print(f'{name}; {closestMin}; {(xmin[0][0], xmin[1][0])}; {f}; {neval}; {minDist}')
        file.write(f'{name}; {closestMin}; {(xmin[0][0], xmin[1][0])}; {f}; {neval}; {minDist}\n')
        # draw(coords, xmin, len(coords), func, flag, lb, ub)
        
if __name__ == '__main__':
    #accuracyByAgentsAndIterations2d()
    main()
    #accuracyByAgentsAndIterations()
    #faccuracyByAgentsAndIterations()
