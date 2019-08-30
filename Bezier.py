import numpy as np
from scipy.special import comb


class Bezier:

    def __init__(self, points):
        self.controlPoints = points
        self.degree = len(points)
        self.culBernsteinMatrix()

    def culBernsteinMatrix(self):
        self.matBernstein = np.array(np.zeros((self.degree, self.degree)))
        for col in range(self.degree):
            Coef = comb(self.degree - 1, col)
            for row in range(col, self.degree):
                curCoef = Coef
                if ((row + col) % 2 != 0):
                    curCoef = -Coef
                self.matBernstein[row, col] = curCoef * comb(self.degree - col - 1, row - col)

    def samplePoint(self, v):
        vecParam = np.array(np.zeros((self.degree, 1)))
        for i in range(self.degree):
            vecParam[i] = pow(v, i)
        return np.dot(np.dot(vecParam.transpose(), self.matBernstein), self.controlPoints)

    def getTangent(self, v):
        vecParam = np.zeros((self.degree, 1))
        for i in range(self.degree):
            vecParam[i] = i * pow(v, i - 1)
        return np.dot(np.dot(vecParam.transpose(), self.matBernstein), self.controlPoints)

    def getNormal(self, v):
        tangent = self.getTangent(v)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        return np.dot(rotation, tangent.transpose()).transpose()

