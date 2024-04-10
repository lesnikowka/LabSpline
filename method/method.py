import numpy as np
import matplotlib.pyplot as plt


# Тесотовая функция
def tf(x):
    if x >= -1 and x <= 0:
        return x ** 3 + 3 * x ** 2
    return -x ** 3 + 3 * x ** 2


def dtf(x):
    if x >= -1 and x <= 0:
        return 3 * x ** 2 + 6 * x
    return -3 * x ** 2 + 6 * x


def d2tf(x):
    if x >= -1 and x <= 0:
        return 6 * x + 6
    return -6 * x + 6


# Функции для основной задачи 1
def Task1Func1(x):
    return np.log(x + 1) / (x + 1)


# def dTask1Func1(x): #Производная первой функции
#    return (x + (-x - 1) * np.log(x + 1)) / (x ** 3 + x ** 2)

def dTask1Func1(x):  # Производная первой функции
    return (1 - np.log(x + 1)) / (x + 1) ** 2


# def d2Task1Func1(x): #Вторая производная первой функции
#    return (-np.log(x + 1) * (x ** 3 + x ** 2) - 3 * x ** 3 - 2 * x ** 2 - 3 * np.log(x + 1) * x ** 2 * (-x - 1) - 2 * np.log(x + 1) * x * (-x - 1)) / (x ** 3 + x ** 2) ** 2

def d2Task1Func1(x):
    return (-2 * (x + 1) * (1 - np.log(x + 1)) - x - 1) / (x + 1) ** 4


def Task1Func2(x):
    return (x ** 0.5) * np.sin(x)


def dTask1Func2(x):  # Производная второй функции
    return (np.sin(x) + 2 * x * np.cos(x)) / (2 * x ** 0.5)


def d2Task1Func2(x):  # Вторая производная второй функции
    return (4 * x * np.cos(x) - 4 * x ** 2 * np.sin(x) - np.sin(x)) / (4 * x * x ** 0.5)


# Функции для основной задачи 2
def Task2Func1(x):
    return np.log(x + 1) / (x + 1) + np.cos(10*x)


def dTask2Func1(x):
    return dTask1Func1(x) - 10*np.sin(10*x)


def d2Task2Func1(x):
    return d2Task1Func1(x) - 100*np.cos(10*x)


def Task2Func2(x):
    return Task1Func2(x) + np.cos(10*x)


def dTask2Func2(x):
    return dTask1Func2(x) - 10*np.sin(10*x)


def d2Task2Func2(x):
    return d2Task1Func2(x) - 100*np.cos(10*x)


def initCoeff(n, h, left, right, fval):
    Ai = h
    Bi = h
    Ci = -4 * h
    kapa1 = 0
    kapa2 = 0
    mu1 = left
    mu2 = right
    phi = [0] * (n - 1)
    for i in range(0, n - 1):
        phi[i] = - 6 * ((fval[i + 2] - 2 * fval[i + 1] + fval[i]) / h)
    coeffs = [kapa1, kapa2, mu1, mu2, phi, Ai, Bi, Ci]
    return coeffs


# Прогонка-вычисления j-го слоя
def runTrough(coeffs, n):
    v = [0] * (n + 1)
    kapa1 = coeffs[0]
    kapa2 = coeffs[1]
    mu1 = coeffs[2]
    mu2 = coeffs[3]
    phi = coeffs[4]
    Ai = coeffs[5]
    Bi = coeffs[6]
    Ci = coeffs[7]

    # Прямой ход
    alpha = [0] * (n)
    betta = [0] * (n)
    alpha[0] = kapa1
    betta[0] = mu1
    for i in range(1, n):
        alpha[i] = Bi / (Ci - Ai * alpha[i - 1])
        betta[i] = (phi[i - 1] + Ai * betta[i - 1]) / (Ci - Ai * alpha[i - 1])

    # Обратный ход
    v[n] = (-kapa2 * betta[n - 1] - mu2) / (kapa2 * alpha[n - 1] - 1)
    for i in range(n - 1, -1, -1):
        v[i] = alpha[i] * v[i + 1] + betta[i]
    return v


# Вычисление коэфициентов сплайна
def splainCoef(a, b, n, leftBound, rightBound, f):
    h = abs((a - b) / n)
    x = np.linspace(a, b, n + 1)
    funcVal = []
    for i in range(n + 1):
        funcVal.append(f(x[i]))

    coef = initCoeff(n, h, leftBound, rightBound, funcVal)
    C = runTrough(coef, n)

    A = funcVal[1::]

    D = []
    for i in range(1, n + 1):
        D.append((C[i] - C[i - 1]) / h)

    B = []
    for i in range(1, n + 1):
        B.append((funcVal[i] - funcVal[i - 1]) / h + C[i] * h / 3 + C[i - 1] * h / 6)

    return (A, B, C[1::], D, x)


def splain(A, B, C, D, X, n, x):
    for i in range(1, len(X)):
        if x >= X[i - 1] and x < X[i]:
            return A[i - 1] + B[i - 1] * (x - X[i]) + (C[i - 1] / 2) * ((x - X[i]) ** 2) + (D[i - 1] / 6) * (
                    (x - X[i]) ** 3)
        elif x == X[i]:
            return A[i - 1]


def splainDer(a, b, c, d):
    a_ = a.copy()
    b_ = b.copy()
    c_ = c.copy()
    d_ = d.copy()
    for i in range(len(a_)):
        a_[i] = b_[i]
        b_[i] = c_[i]
        c_[i] = d_[i]
        d_[i] = 0
    return a_, b_, c_, d_


def calculate(N, A, B, a_, b_, func, funcd, func2d):
    a, b, c, d, xval = splainCoef(A, B, N, a_, b_, func)
    a_, b_, c_, d_ = splainDer(a, b, c, d)
    a__, b__, c__, d__ = splainDer(a_, b_, c_, d_)
    xx = np.linspace(A, B, 2 * N + 1)
    y = [splain(a, b, c, d, xval, N, x) for x in xx]
    yy = [func(x) for x in xx]
    y_ = [splain(a_, b_, c_, d_, xval, N, x) for x in xx]
    yy_ = [funcd(x) for x in xx]
    y__ = [splain(a__, b__, c__, d__, xval, N, x) for x in xx]
    yy__ = [func2d(x) for x in xx]

    return xx, [a, b, c, d], [y, yy], [y_, yy_], [y__, yy__], xval


def calculateTf(N, A, B, a_, b_):
    return calculate(N, A, B, a_, b_, tf, dtf, d2tf)


def calculateMain11(N, A, B, a_, b_):
    return calculate(N, A, B, a_, b_, Task1Func1, dTask1Func1, d2Task1Func1)


def calculateMain12(N, A, B, a_, b_):
    return calculate(N, A, B, a_, b_, Task1Func2, dTask1Func2, d2Task1Func2)


def calculateMain21(N, A, B, a_, b_):
    return calculate(N, A, B, a_, b_, Task2Func1, dTask2Func1, d2Task2Func1)


def calculateMain22(N, A, B, a_, b_):
    return calculate(N, A, B, a_, b_, Task2Func2, dTask2Func2, d2Task2Func2)
