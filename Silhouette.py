import numpy as np
from PcdReconstruction.Bezier import Bezier


class Silhouette(Bezier):
    def __init__(self, Midrib, Params):
        self.midrib = Midrib
        self.params = Params
        self.setControlPoints()
        self.culBernsteinMatrix()

    def setControlPoints(self):
        self.controlPoints = np.array([[0, 0, 0]])
        self.degree = len(self.params) + 2
        self.controlPoints[0] = self.midrib.controlPoints[0]
        for i in range(len(self.params)):
            A = self.midrib.samplePoint(self.params[i, 0])
            N = self.midrib.getNormal(self.params[i, 0])
            self.controlPoints = np.append(self.controlPoints, A + self.params[i, 1] * N + self.params[i, 2] * np.array([[0, 0, 1]]), axis=0)
        self.controlPoints = np.append(self.controlPoints, self.midrib.controlPoints[-1].reshape(1,3), axis=0)
