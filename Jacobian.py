from sympy import *
import numpy as np
from scipy.special import comb

x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21 = symbols(
    "x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21")


def culBernsteinMatrix(degree):
    matBernstein = np.array(np.zeros((degree, degree)))
    for col in range(degree):
        Coef = comb(degree - 1, col)
        for row in range(col, degree):
            curCoef = Coef
            if ((row + col) % 2 != 0):
                curCoef = -Coef
            matBernstein[row, col] = curCoef * comb(degree - col - 1, row - col)

    return matBernstein


def A(v):
    vecParam = Matrix([[1, v, v ** 2, v ** 3]])
    coef = Matrix(culBernsteinMatrix(4))
    ctrlP = Matrix([[0, 0, 0], [x1, x2, x3], [x4, x5, x6], [0, 1, 0]])
    return vecParam * coef * ctrlP


def N(v):
    vecParam = Matrix([[0, 1, 2 * v, 3 * v ** 2]])
    coef = Matrix(culBernsteinMatrix(4))
    ctrlP = Matrix([[0, 0, 0], [x1, x2, x3], [x4, x5, x6], [0, 1, 0]])
    tangent = vecParam * coef * ctrlP
    rotation = Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    return (rotation * tangent.T).T


def S(v):
    S1 = A(x7) + x8 * N(x7) + Matrix([[0, 0, x9]])
    S2 = A(x10) + x11 * N(x10) + Matrix([[0, 0, x12]])
    S3 = A(x13) + x14 * N(x13) + Matrix([[0, 0, x15]])

    vecParam = Matrix([[1, v, v ** 2, v ** 3, v ** 4]])
    coef = Matrix(culBernsteinMatrix(5))
    ctrlP = Matrix([[0, 0, 0], S1, S2, S3, [0, 1, 0]])

    return vecParam * coef * ctrlP


def F(v):
    F1 = A(x7) + (1 / 3) * x8 * N(x7) + Matrix([[0, 0, x16]])
    F2 = A(x10) + (1 / 3) * x11 * N(x10) + Matrix([[0, 0, x17]])
    F3 = A(x13) + (1 / 3) * x14 * N(x13) + Matrix([[0, 0, x18]])

    vecParam = Matrix([[1, v, v ** 2, v ** 3, v ** 4]])
    coef = Matrix(culBernsteinMatrix(5))
    ctrlP = Matrix([[0, 0, 0], F1, F2, F3, [0, 1, 0]])

    return vecParam * coef * ctrlP


def G(v):
    G1 = A(x7) + (2 / 3) * x8 * N(x7) + Matrix([[0, 0, x19]])
    G2 = A(x10) + (2 / 3) * x11 * N(x10) + Matrix([[0, 0, x20]])
    G3 = A(x13) + (2 / 3) * x14 * N(x13) + Matrix([[0, 0, x21]])

    vecParam = Matrix([[1, v, v ** 2, v ** 3, v ** 4]])
    coef = Matrix(culBernsteinMatrix(5))
    ctrlP = Matrix([[0, 0, 0], G1, G2, G3, [0, 1, 0]])

    return vecParam * coef * ctrlP


def L(u, v):
    vecParam = Matrix([[1, u, u ** 2, u ** 3]])
    coef = Matrix(culBernsteinMatrix(4))
    ctrlP = Matrix([A(v), F(v), G(v), S(v)])

    return vecParam * coef * ctrlP

