import numpy as np
import matplotlib.pyplot as plt

# функция инициализации агентов поиска
def initialization_SAO(SearchAgents_no, dim, boundUp, boundLow):
    Boundary_no = len(boundUp)
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (boundUp - boundLow) + boundLow
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            Positions[:, i] = np.random.rand(SearchAgents_no) * (boundUp[i] - boundLow[i]) + boundLow[i]#i-ая колонка
    return Positions

# функция, реализующая алгоритм SAO
def SAO(N, Max_iter, boundLow, boundUp, dim, fobj):
    if np.isscalar(boundUp): 
        boundUp = np.full(dim, boundUp) 
        boundLow = np.full(dim, boundLow)

    X = initialization_SAO(N, dim, boundUp, boundLow)
    Best_pos = np.zeros(dim) 
    Best_score = float('inf') 

    Objective_values = np.zeros(N)
    steps = []
    N1 = N // 2 #первая подпопуляция

    for i in range(N):
        Objective_values[i] = fobj(X[i, :])
        if Objective_values[i] < Best_score:
            Best_score = Objective_values[i]
            Best_pos = X[i, :].copy()

    sorted_indices = np.argsort(Objective_values) #отсортированные индексы
    second_best = X[sorted_indices[1], :] #второй лучший индивид в популяции       
    third_best = X[sorted_indices[2], :]  #третий       
    half_best_mean = np.mean(X[sorted_indices[:N1], :], axis=0) #центроидное положение

    Elite_pool = [Best_pos, second_best, third_best, half_best_mean] #список лучших особей
    best1 = np.array([[Best_pos[0]],[Best_pos[1]]])
    steps.append(best1)
    for l in range(2, Max_iter + 1): 
        RB = np.random.randn(N, dim) #используем нормальное распределение, где
                                     #N - кол-во строк в матрице, dim - кол-во столбцов
        
        T = np.exp(-l / Max_iter) #средняя дневная температура
        DDF = 0.35 * (1 + (5 / 7) * (np.exp(l / Max_iter) - 1) / (np.exp(1) - 1)) #коэффициент градусо-дней
        M = DDF * T #подсчет градусодней

        X_centroid = np.mean(X, axis=0) #центроидное положение индивидуумов
        indices = np.arange(N)
        Na = max(1, N // 2) 

        index1 = np.random.choice(N, Na, replace=False) 
        index2 = np.setdiff1d(indices, index1)

        #обновление позиции объектов 
        for i in index1:
            r1 = np.random.rand() 
            k1 = np.random.randint(0, 4) 
            X[i, :] = Elite_pool[k1] + RB[i, :] * (
                r1 * (Best_pos - X[i, :]) + (1 - r1) * (X_centroid - X[i, :])
            )
        #выбор элитного значения

        #обновление позиции объектов
        for i in index2:
            r2 = 2 * np.random.rand() - 1
            X[i, :] = M * Best_pos + RB[i, :] * (
                r2 * (Best_pos - X[i, :]) + (1 - r2) * (X_centroid - X[i, :])
            )

        X = np.clip(X, boundLow, boundUp) #ограничение значений в диапазоне [bL, bU]

        #поиск наилучшего значения
        for i in range(N):
            Objective_values[i] = fobj(X[i, :])
            if Objective_values[i] < Best_score: 
                Best_score = Objective_values[i] #значение целевой функции
                Best_pos = X[i, :].copy() #новая текущая позиция

        sorted_indices = np.argsort(Objective_values)
        second_best = X[sorted_indices[1], :] #вторая лучшая позиция 
        third_best = X[sorted_indices[2], :]  #третья лучшая позиция 
        half_best_mean = np.mean(X[sorted_indices[:N1], :], axis=0) #среднее значение для трех позиций

        Elite_pool = [Best_pos, second_best, third_best, half_best_mean] 
        best1 = np.array([[Best_pos[0]],[Best_pos[1]]])
        steps.append(best1)

    best1 = np.array([[Best_pos[0]],[Best_pos[1]]])
    xmin = best1
    fmin = fobj(Best_pos)
    neval = len(steps)
    coords = steps
    answer_ = [xmin, fmin, neval, coords]
    
    return answer_
