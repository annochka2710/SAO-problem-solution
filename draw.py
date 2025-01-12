import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import *
import random


def contourPlot(ax, f, LimMin, LimMax):
    # Подготовка к рисованию, настраиваем оси x и y
    x1 = np.arange( LimMin, LimMax + 0.1, 0.1)
    m = len(x1)
    y1 = np.arange( LimMin, LimMax + 0.1, 0.1)
    n = len(y1)

    # делаем сетку
    [xx, yy] = np.meshgrid(x1, y1)

    # массивы для графиков функции и ее производных по x и y
    F = np.zeros((n, m))

    # вычисляем рельеф поверхности
    for i in range(n):
        for j in range(m):
            X = [xx[i, j], yy[i, j]]
            F[i, j] = f(X)

    nlevels = 1000
    ax.contour(xx, yy, F, nlevels, linewidths=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


#   - если не задавать цвет, то на итоговом графике видны шаги и маркер выглядит тогда лишним
# из минуса - нет возможности приближать график
def prDraw(ax, coords, nsteps, xmin):
    fSize = 11
    x0 = coords[0]
    for i in range(nsteps - 1):
        x0 = coords[i]
        x1 = coords[i + 1]
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='.', linestyle='solid', ms=3)  # plot line

    ax.text(xmin[0] + 0.35, xmin[1] + 0.2, str(nsteps), fontsize=fSize)
    ax.scatter(xmin[0], xmin[1], marker='o', c='red', s=5, zorder=12)


def draw(coords, xmin, nsteps, f, flag, LimMin, LimMax):
    fig, ax = plt.subplots()
    fig.suptitle('Optimization visualisation & Countour plot')
    plt.xlim(LimMin, LimMax)
    plt.ylim(LimMin, LimMax)
    plt.gca().set_aspect('equal', adjustable='box')
    contourPlot(ax, f, LimMin, LimMax)
    prDraw(ax, coords, nsteps, xmin)
    name = "plot" + flag + ".png"
    fig.savefig(name, dpi=600)
    ad = "<img width=\"1500px\" src=\"/resources/" + name + "\">"
    print(ad)