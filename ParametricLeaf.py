from PcdReconstruction.Bezier import Bezier
from PcdReconstruction.Midrib import Midrib
from PcdReconstruction.Silhouette import Silhouette
import numpy as np

MIDRIB_SAMPLE = 50
CROSS_SECTION_SAMPLE = 20


class ParametricLeaf:
    def __init__(self):
        self.A1 = np.array([0, 0.3, 0])
        self.A2 = np.array([0, 0.7, 0])
        self.S1 = np.array([0.1, 0.25, 0])
        self.S2 = np.array([0.5, 0.4, 0])
        self.S3 = np.array([0.9, 0.25, 0])
        self.F = np.array([0.0, 0.0, 0.0])
        self.G = np.array([0.0, 0.0, 0.0])

    def computeParamLeafModel(self):
        self.computeMidrib()
        self.computeSilhouette()
        self.computeLongitudinal()
        self.computeCrossSection()

    def computeMidrib(self):
        self.midrib = Midrib(np.array([[0, 0, 0], self.A1, self.A2, [0, 1, 0]]))

    def computeSilhouette(self):
        self.silhouetteL = Silhouette(self.midrib, np.array([self.S1, self.S2, self.S3]))
        self.silhouetteR = Silhouette(self.midrib,
                                      np.array([np.multiply(self.S1, [1, -1, 1]), np.multiply(self.S2, [1, -1, 1]),
                                                np.multiply(self.S3, [1, -1, 1])]))

    def computeLongitudinal(self):
        self.curveFL = Silhouette(self.midrib, np.array(
            [[self.S1[0], self.S1[1] * (1 / 3), self.F[0]],
             [self.S2[0], self.S2[1] * (1 / 3), self.F[1]],
             [self.S3[0], self.S3[1] * (1 / 3), self.F[2]]]))
        self.curveFR = Silhouette(self.midrib, np.array(
            [[self.S1[0], self.S1[1] * (-1 / 3), self.F[0]],
             [self.S2[0], self.S2[1] * (-1 / 3), self.F[1]],
             [self.S3[0], self.S3[1] * (-1 / 3), self.F[2]]]))
        self.curveGL = Silhouette(self.midrib, np.array(
            [[self.S1[0], self.S1[1] * (2 / 3), self.G[0]],
             [self.S2[0], self.S2[1] * (2 / 3), self.G[1]],
             [self.S3[0], self.S3[1] * (2 / 3), self.G[2]]]))
        self.curveGR = Silhouette(self.midrib, np.array(
            [[self.S1[0], self.S1[1] * (-2 / 3), self.G[0]],
             [self.S2[0], self.S2[1] * (-2 / 3), self.G[1]],
             [self.S3[0], self.S3[1] * (-2 / 3), self.G[2]]]))

    def getCrossSection(self, v, direction):
        cv0 = self.midrib.samplePoint(v).flatten()
        if direction == -1:
            cv1 = self.curveFL.samplePoint(v).flatten()
            cv2 = self.curveGL.samplePoint(v).flatten()
            cv3 = self.silhouetteL.samplePoint(v).flatten()
        elif direction == 1:
            cv1 = self.curveFR.samplePoint(v).flatten()
            cv2 = self.curveGR.samplePoint(v).flatten()
            cv3 = self.silhouetteR.samplePoint(v).flatten()

        return Bezier(np.array([cv0, cv1, cv2, cv3]))

    def computeCrossSection(self):
        self.vertices = []

        self.vertices.append([0, 0, 0])
        for i in range(1, MIDRIB_SAMPLE):
            v = (1 / MIDRIB_SAMPLE) * i
            self.crossSectionL = self.getCrossSection(v, -1)
            gap = 1 / CROSS_SECTION_SAMPLE
            for j in range(CROSS_SECTION_SAMPLE, -1, -1):
                self.vertices.append(self.crossSectionL.samplePoint(gap * j).flatten().tolist())

            self.crossSectionR = self.getCrossSection(v, 1)
            for j in range(1, CROSS_SECTION_SAMPLE + 1):
                self.vertices.append(self.crossSectionR.samplePoint(gap * j).flatten().tolist())

        self.vertices.append([0, 1, 0])

        self.face = []
        self.normal = []

        num = len(self.vertices)
        for i in range(-1, MIDRIB_SAMPLE - 1):
            if i == -1:
                for j in range(0, 2 * CROSS_SECTION_SAMPLE):
                    self.face.append([0, j + 2, j + 1])
                    A = np.array(self.vertices[0])
                    B = np.array(self.vertices[j + 2])
                    C = np.array(self.vertices[j + 1])
                    self.normal.append(np.cross((B - A), (C - A)).flatten().tolist())
                continue

            if i == MIDRIB_SAMPLE - 2:
                for j in range(0, 2 * CROSS_SECTION_SAMPLE):
                    self.face.append([num - 1,
                                      i * (2 * CROSS_SECTION_SAMPLE + 1) + j + 1,
                                      i * (2 * CROSS_SECTION_SAMPLE + 1) + j + 2])
                    A = np.array(self.vertices[num - 1])
                    B = np.array(self.vertices[i * (2 * CROSS_SECTION_SAMPLE + 1) + j + 1])
                    C = np.array(self.vertices[i * (2 * CROSS_SECTION_SAMPLE + 1) + j + 2])
                    self.normal.append(np.cross((B - A), (C - A)).flatten().tolist())
                continue

            for j in range(1, 2 * CROSS_SECTION_SAMPLE + 1):
                self.face.append([i * (2 * CROSS_SECTION_SAMPLE + 1) + j,
                                  i * (2 * CROSS_SECTION_SAMPLE + 1) + j + 1,
                                  (i + 1) * (2 * CROSS_SECTION_SAMPLE + 1) + j])
                A = np.array(self.vertices[(i + 1) * (2 * CROSS_SECTION_SAMPLE + 1) + j])
                B = np.array(self.vertices[i * (2 * CROSS_SECTION_SAMPLE + 1) + j])
                C = np.array(self.vertices[i * (2 * CROSS_SECTION_SAMPLE + 1) + j + 1])
                self.normal.append(np.cross((B - A), (C - A)).flatten().tolist())

                self.face.append([i * (2 * CROSS_SECTION_SAMPLE + 1) + j + 1,
                                  (i + 1) * (2 * CROSS_SECTION_SAMPLE + 1) + j + 1,
                                  (i + 1) * (2 * CROSS_SECTION_SAMPLE + 1) + j])
                A = np.array(self.vertices[i * (2 * CROSS_SECTION_SAMPLE + 1) + j + 1])
                B = np.array(self.vertices[(i + 1) * (2 * CROSS_SECTION_SAMPLE + 1) + j + 1])
                C = np.array(self.vertices[(i + 1) * (2 * CROSS_SECTION_SAMPLE + 1) + j])
                self.normal.append(np.cross((B - A), (C - A)).flatten().tolist())
