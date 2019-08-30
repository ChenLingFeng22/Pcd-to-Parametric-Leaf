from OpenGL.GL import *
from OpenGL.GLU import *
from math import *
from sklearn import preprocessing
import numpy as np

PI = 3.1415926535897


class Camera:
    def __init__(self, camera=None):
        if camera == None:
            self.projNear = 0.1
            self.projFar = 1000
            self.eye = np.array([[0, 0, 0]])
            self.at = np.array([[0, 0, 0]])
            self.upVec = np.array([[0, 0, 0]])
            self.fov = 45
            self.distance = 1
            self.widgetSize = [0, 0]
        else:
            self.setCamera(camera)

    def setCamera(self, camera):
        self.projNear = camera.projNear
        self.projFar = camera.projFar
        self.eye = camera.eye
        self.at = camera.at
        self.upVec = camera.upVec
        self.fov = camera.fov
        self.distance = camera.distance
        self.widgetSize = camera.widgetSize

    def draw(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(self.eye[0, 0], self.eye[0, 1], self.eye[0, 2],
                  self.at[0, 0], self.at[0, 1], self.at[0, 2],
                  self.upVec[0, 0], self.upVec[0, 1], self.upVec[0, 2])

    def getFarward(self):
        return preprocessing.normalize(self.at - self.eye, norm='l2')

    def getRight(self):
        return preprocessing.normalize(np.cross(self.getFarward(), self.upVec), norm='l2')

    def getHorizonFov(self):
        return atan(self.widgetSize[0] * tan(self.fov / 2) / self.widgetSize[1]) * 360 / PI

    def pan(self, dx, dy):
        up = self.upVec
        right = self.getRight()
        move = (up * (dy) + right * (-dx)) * 0.014 * self.distance

        self.eye = self.eye + move
        self.at = self.at + move

    def orbit(self, dx, dy):
        dist = np.linalg.norm(self.at - self.eye, ord=2)
        up = preprocessing.normalize(np.cross(self.getRight(), self.getFarward()), norm='l2')

        right = self.getRight()
        move = (up * (dy) + right * (-dx)) * 0.07 * self.distance
        self.eye = self.eye + move

        direction = preprocessing.normalize(np.array(self.eye) - np.array(self.at), norm='l2')
        self.eye = self.at + direction * dist

    def zoom(self, d):
        self.farward = self.getFarward()
        move = (self.farward * d) * 0.001 * self.distance
        self.eye = self.eye + move

    def updateWindowSize(self, width, height):
        glViewport(0, 0, width, height)
        aspect = width / height
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, aspect, self.projNear, self.projFar)
        self.widgetSize = [width, height]
