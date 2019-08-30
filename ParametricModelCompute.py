from PcdReconstruction.ParametricLeaf import *
from PcdReconstruction.Jacobian import *
from sklearn.decomposition import PCA
from scipy.optimize import fminbound
from scipy.optimize import minimize
import numpy as np
import datetime
import math

#扫描线数量
MIDRIB_SAMPLE = 50
#参数文件
paraFilename = 'temp.param'
#点云文件
Pcdfilename = 'leafModel/leaf_1.pcd'

#扫描线
class scanline:
    def __init__(self, points, Center_point):
        self.Center_point = Center_point
        self.lengthL, self.lengthR = self.getLength(points)

    def getLength(self, points):
        if len(points) > 0:
            lengthL = points[0].u
            lengthR = points[0].u
            for i in range(len(points)):
                if points[i].u < 0:
                    if lengthL < abs(points[i].u):
                        lengthL = abs(points[i].u)
                elif points[i].u > 0:
                    if lengthR < abs(points[i].u):
                        lengthR = abs(points[i].u)

            return lengthL, lengthR

        else:
            return 0, 0

#点云对应的uv坐标
class UV_point:
    def __init__(self, point_index, u, v):
        self.point_index = point_index
        self.u = u
        self.v = v


class PcdToParaLeaf:
    def __init__(self, points, paraLeaf):
        self.iteration_num = 0
        self.paraLeaf = paraLeaf
        #PCA降维
        self.originalPoints = points
        pca = PCA(n_components=3)
        self.points = np.array(pca.fit_transform(self.originalPoints))
        self.points = np.column_stack((self.points, np.ones(self.points.shape[0])))
        #获取apex点和base点
        self.apex_index = np.argmin(self.points, axis=0)[0]
        self.base_index = np.argmax(self.points, axis=0)[0]
        # 对点进行刚性变换
        self.rigidTransform()
        self.points = np.delete(self.points, -1, axis=1)

    #获取旋转角度
    def getAngle(self, x, y):
        angle = math.atan2(abs(x), abs(y))
        if x > 0 and y > 0:
            return -angle
        elif x < 0 and y > 0:
            return angle
        elif x < 0 and y < 0:
            return math.pi - angle
        elif x > 0 and y < 0:
            return -(math.pi - angle)

    #对点云进行刚性变换
    def rigidTransform(self):
        translate = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
             [-self.points[self.base_index][0], -self.points[self.base_index][1], 0, 1]])
        self.points = np.dot(self.points, translate)

        angle = self.getAngle(self.points[self.apex_index][0], self.points[self.apex_index][1])
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0, 0],
                             [math.sin(angle), math.cos(angle), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        self.points = np.dot(self.points, rotation)


        scale = np.array([[1 / self.points[self.apex_index][1], 0, 0, 0],
                          [0, 1 / self.points[self.apex_index][1], 0, 0],
                          [0, 0, 1 / self.points[self.apex_index][1], 0],
                          [0, 0, 0, 1]])
        self.points = np.dot(self.points, scale)

    #计算点到扫描线的距离
    def distance_P_to_L(self, N, L_point, point):
        v = point - L_point
        v1 = np.array([v[0][0], v[0][1]])
        v2 = np.array([N[0][0], N[0][1]])
        return np.linalg.norm(np.cross(v1, v2) / np.linalg.norm(v2))

    #计算点到中轴线的距离
    def distance_P_to_Midrid(self, point):
        def D(u):
            return np.linalg.norm(self.paraLeaf.midrib.samplePoint(u) - point)

        u = fminbound(D, 0, 1)
        return D(u)

    #将点云从笛卡尔坐标系转换到uv坐标系
    def UVmapping(self):
        self.scanlines = []
        self.UV_points = []
        pointEnter = [False] * len(self.points)
        stride = 1 / MIDRIB_SAMPLE
        for i in range(1, MIDRIB_SAMPLE):
            # 中轴线的法向量
            N = self.paraLeaf.midrib.getNormal(stride * i)
            # 中轴线上的点转化为笛卡尔坐标系
            L_point = self.paraLeaf.midrib.samplePoint(stride * i)
            tempUVpoints = []
            if i == MIDRIB_SAMPLE - 1:
                p1 = np.array([self.paraLeaf.midrib.samplePoint(i * stride)[0][0],
                               self.paraLeaf.midrib.samplePoint(i * stride)[0][1]])
                p2 = np.array([self.paraLeaf.midrib.samplePoint((i - 1) * stride)[0][0],
                               self.paraLeaf.midrib.samplePoint((i - 1) * stride)[0][1]])
                delta = np.linalg.norm(p1 - p2)
            else:
                p1 = np.array([self.paraLeaf.midrib.samplePoint(i * stride)[0][0],
                               self.paraLeaf.midrib.samplePoint(i * stride)[0][1]])
                p2 = np.array([self.paraLeaf.midrib.samplePoint((i + 1) * stride)[0][0],
                               self.paraLeaf.midrib.samplePoint((i + 1) * stride)[0][1]])
                delta = np.linalg.norm(p1 - p2)

            for j in range(len(self.points)):
                if pointEnter[j] == False:
                    distance = self.distance_P_to_L(N, L_point, self.points[j])
                    # 如果点到scanline的距离小于 delta/2, 则点的v等于L_point的v，u为点到Midrib的欧式距离
                    if distance <= (delta / 2):
                        if self.points[j][0] <= L_point[0][0]:
                            uv_point = UV_point(j, -self.distance_P_to_Midrid(self.points[j]), i * stride)

                        elif self.points[j][0] > L_point[0][0]:
                            uv_point = UV_point(j, self.distance_P_to_Midrid(self.points[j]), i * stride)

                        tempUVpoints.append(uv_point)
                        pointEnter[j] = [True]

            sline = scanline(tempUVpoints, L_point)
            self.scanlines.append(sline)

            # u坐标归一化
            for i in range(len(tempUVpoints)):
                if tempUVpoints[i].u < 0:
                    tempUVpoints[i].u /= sline.lengthL
                elif tempUVpoints[i].u > 0:
                    tempUVpoints[i].u /= sline.lengthR

            self.UV_points.extend(tempUVpoints)

    # 损失函数
    def Loss(self):
        loss = 0
        for i in range(len(self.UV_points)):
            v = self.UV_points[i].v
            u = self.UV_points[i].u
            if u != 0:
                Luv = self.paraLeaf.getCrossSection(v, u / abs(u)).samplePoint(abs(u))
            else:
                Luv = self.paraLeaf.midrib.samplePoint(v)

            Puv = self.points[self.UV_points[i].point_index]
            loss += np.linalg.norm(Luv - Puv)

        return loss

    def func(self, x):
        self.paraLeaf.A1 = np.array([x[0], x[1], x[2]])
        self.paraLeaf.A2 = np.array([x[3], x[4], x[5]])
        self.paraLeaf.S1 = np.array([x[6], x[7], x[8]])
        self.paraLeaf.S2 = np.array([x[9], x[10], x[11]])
        self.paraLeaf.S3 = np.array([x[12], x[13], x[14]])
        self.paraLeaf.F = np.array([x[15], x[16], x[17]])
        self.paraLeaf.G = np.array([x[18], x[19], x[20]])
        self.paraLeaf.computeParamLeafModel()

        ParaWriter(paraFilename, self.paraLeaf)

        self.UVmapping()
        loss = self.Loss()

        self.iteration_num += 1

        print('parametric: ')
        print(
            [self.paraLeaf.A1, self.paraLeaf.A2, self.paraLeaf.S1, self.paraLeaf.S2, self.paraLeaf.S3, self.paraLeaf.F,
             self.paraLeaf.G])
        print('loss: ', loss)
        print('iteration: ', self.iteration_num)

        return loss

    #参数的约束
    def con(self):
        #  -0.5 < A1.x, A2.x < 0.5     0 < A1.y, A2.y < 1     -0.5 < A1.z, A2.z < 0.5
        #  -0.5 < S1.v < 0.5,   0 < S2.v S3.v < 1   0 < S1.x, S2.x S3.x < 1   -0.5 < S1.z, S2.z S3.z < 0.5
        #  -0.1 < F.z1, F.z2 F.z3 < 0.1   -0.1 < G.z1, G.z2 G.z3 < 0.1
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - (-0.5)}, \
                {'type': 'ineq', 'fun': lambda x: -x[0] + 0.5}, \
                {'type': 'ineq', 'fun': lambda x: x[1]}, \
                {'type': 'ineq', 'fun': lambda x: -x[1] + 1}, \
                {'type': 'ineq', 'fun': lambda x: x[2] - (-0.5)}, \
                {'type': 'ineq', 'fun': lambda x: -x[2] + 0.5}, \
                {'type': 'ineq', 'fun': lambda x: x[3] - (-0.5)}, \
                {'type': 'ineq', 'fun': lambda x: -x[3] + 0.5}, \
                {'type': 'ineq', 'fun': lambda x: x[4]}, \
                {'type': 'ineq', 'fun': lambda x: -x[4] + 1}, \
                {'type': 'ineq', 'fun': lambda x: x[5] - (-0.5)}, \
                {'type': 'ineq', 'fun': lambda x: -x[5] + 0.5}, \
                {'type': 'ineq', 'fun': lambda x: x[6] - (-0.5)}, \
                {'type': 'ineq', 'fun': lambda x: -x[6] + 0.5}, \
                {'type': 'ineq', 'fun': lambda x: x[7]}, \
                {'type': 'ineq', 'fun': lambda x: -x[7] + 1}, \
                {'type': 'ineq', 'fun': lambda x: x[8] - (-0.5)}, \
                {'type': 'ineq', 'fun': lambda x: -x[8] + 0.5}, \
                {'type': 'ineq', 'fun': lambda x: x[9]}, \
                {'type': 'ineq', 'fun': lambda x: -x[9] + 1}, \
                {'type': 'ineq', 'fun': lambda x: x[10]}, \
                {'type': 'ineq', 'fun': lambda x: -x[10] + 1}, \
                {'type': 'ineq', 'fun': lambda x: x[11] - (-0.5)}, \
                {'type': 'ineq', 'fun': lambda x: -x[11] + 0.5}, \
                {'type': 'ineq', 'fun': lambda x: x[12]}, \
                {'type': 'ineq', 'fun': lambda x: -x[12] + 1}, \
                {'type': 'ineq', 'fun': lambda x: x[13]}, \
                {'type': 'ineq', 'fun': lambda x: -x[13] + 1}, \
                {'type': 'ineq', 'fun': lambda x: x[14] - (-0.5)}, \
                {'type': 'ineq', 'fun': lambda x: -x[14] + 0.5}, \
                {'type': 'ineq', 'fun': lambda x: x[15] - (-0.1)}, \
                {'type': 'ineq', 'fun': lambda x: -x[15] + 0.1}, \
                {'type': 'ineq', 'fun': lambda x: x[16] - (-0.1)}, \
                {'type': 'ineq', 'fun': lambda x: -x[16] + 0.1}, \
                {'type': 'ineq', 'fun': lambda x: x[17] - (-0.1)}, \
                {'type': 'ineq', 'fun': lambda x: -x[17] + 0.1}, \
                {'type': 'ineq', 'fun': lambda x: x[18] - (-0.1)}, \
                {'type': 'ineq', 'fun': lambda x: -x[18] + 0.1}, \
                {'type': 'ineq', 'fun': lambda x: x[19] - (-0.1)}, \
                {'type': 'ineq', 'fun': lambda x: -x[19] + 0.1}, \
                {'type': 'ineq', 'fun': lambda x: x[20] - (-0.1)}, \
                {'type': 'ineq', 'fun': lambda x: -x[20] + 0.1})
        return cons

    #计算损失函数的Jacobian Matrix
    def jacobian(self, x):
        loss = Matrix([[0] * 21])
        args = Matrix([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21])

        start1 = datetime.datetime.now()
        for i in range(len(self.UV_points)):
            v = self.UV_points[i].v
            u = self.UV_points[i].u
            if u != 0:
                Luv = L(u, v)
            else:
                Luv = A(v)
            Puv = Matrix([self.points[self.UV_points[i].point_index]])
            loss += 2*(Luv - Puv) * Luv.jacobian(args)
        end1 = datetime.datetime.now()
        print("Loss time：" + str((end1 - start1).seconds) + "seconds")

        return loss.evalf(subs={x1: x[0], x2: x[1], x3: x[2], x4: x[3], x5: x[4], x6: x[5], x7: x[6],
                                x8: x[7], x9: x[8], x10: x[9], x11: x[10], x12: x[11], x13: x[12], x14: x[13],
                                x15: x[14], x16: x[15], x17: x[16], x18: x[17], x19: x[18], x20: x[19], x21: x[20]})

    #优化参数
    def OMEGA(self):
        start = datetime.datetime.now()
        print('start')
        x = np.array([self.paraLeaf.A1, self.paraLeaf.A2,
                      self.paraLeaf.S1, self.paraLeaf.S2, self.paraLeaf.S3,
                      self.paraLeaf.F, self.paraLeaf.G])
        cons = self.con()

        #采用带约束的SLSQP优化
        self.omega = minimize(self.func, x, method='SLSQP', constraints=cons)
        print('end')
        end = datetime.datetime.now()
        print("run time：" + str(((end - start).seconds) / 60) + "mins")
        return self.omega

#读取PCD文件
def PcdRead(filename):
    PcdFile = open(filename, 'r')
    points = list()
    if PcdFile != None:
        line = PcdFile.readline()
        while line:
            line = line.split()
            if line[0] == 'DATA':
                line = PcdFile.readline()
                break
            line = PcdFile.readline()

        while line:
            line = line.split()
            points.append([float(line[0]), float(line[1]), float(line[2])])
            line = PcdFile.readline()

    PcdFile.close()

    return points

#读取参数文件
def ParaReader(filename, leaf):
    PcdFile = open(filename, 'r')
    if PcdFile != None:
        line = PcdFile.readline()
        while line:
            line = line.split()
            if line[0] == 'midrib_1:':
                for i in range(3):
                    line = PcdFile.readline().split()
                    leaf.A1[i] = float(line[0])
            elif line[0] == 'midrib_2:':
                for i in range(3):
                    line = PcdFile.readline().split()
                    leaf.A2[i] = float(line[0])
            elif line[0] == 'silhouette_1:':
                for i in range(3):
                    line = PcdFile.readline().split()
                    leaf.S1[i] = float(line[0])
            elif line[0] == 'silhouette_2:':
                for i in range(3):
                    line = PcdFile.readline().split()
                    leaf.S2[i] = float(line[0])
            elif line[0] == 'silhouette_3:':
                for i in range(3):
                    line = PcdFile.readline().split()
                    leaf.S3[i] = float(line[0])
            elif line[0] == 'curve_f:':
                for i in range(3):
                    line = PcdFile.readline().split()
                    leaf.F[i] = float(line[0])
            elif line[0] == 'curve_g:':
                for i in range(3):
                    line = PcdFile.readline().split()
                    leaf.G[i] = float(line[0])
            line = PcdFile.readline()

    PcdFile.close()

#写入PCD文件
def ParaWriter(filename, leaf):
    f = open(filename, 'w')
    if f != None:
        f.write('midrib_1:' + '\n')
        f.write(str(leaf.A1[0]) + '\n' + str(leaf.A1[1]) + '\n' + str(leaf.A1[2]) + '\n')
        f.write('midrib_2:' + '\n')
        f.write(str(leaf.A2[0]) + '\n' + str(leaf.A2[1]) + '\n' + str(leaf.A2[2]) + '\n')
        f.write('silhouette_1:' + '\n')
        f.write(str(leaf.S1[0]) + '\n' + str(leaf.S1[1]) + '\n' + str(leaf.S1[2]) + '\n')
        f.write('silhouette_2:' + '\n')
        f.write(str(leaf.S2[0]) + '\n' + str(leaf.S2[1]) + '\n' + str(leaf.S2[2]) + '\n')
        f.write('silhouette_3:' + '\n')
        f.write(str(leaf.S3[0]) + '\n' + str(leaf.S3[1]) + '\n' + str(leaf.S3[2]) + '\n')
        f.write('curve_f:' + '\n')
        f.write(str(leaf.F[0]) + '\n' + str(leaf.F[1]) + '\n' + str(leaf.F[2]) + '\n')
        f.write('curve_g:' + '\n')
        f.write(str(leaf.G[0]) + '\n' + str(leaf.G[1]) + '\n' + str(leaf.G[2]) + '\n')

    f.close()


if __name__ == '__main__':
    leaf = ParametricLeaf()
    leaf.computeParamLeafModel()
    pcdLeaf = PcdToParaLeaf(PcdRead(Pcdfilename), leaf)
    pcdLeaf.OMEGA()

