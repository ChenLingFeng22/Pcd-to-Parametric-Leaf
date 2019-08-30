from PcdReconstruction.ParametricModelCompute import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from PcdReconstruction.Camera import Camera
import time
import numpy as np
import sys

GLUT_WHEEL_UP = 3
GLUT_WHEEL_DOWN = 4

type = 2
paraFilename = 'temp.param'
Pcdfilename = 'leafModel/leaf_1.pcd'


class Leaf_Render:
    drawType = 0

    def __init__(self, points, leaf):
        self.points = points
        self.leaf = leaf
        self.camera = Camera()

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutInitWindowPosition(100, 100)
        glutInitWindowSize(600, 600)
        glutCreateWindow(b"ParametricLeaf")
        glClearColor(0.95, 0.95, 0.95, 1)

        glutMouseFunc(self.mouseClick)
        glutMotionFunc(self.mouseMove)
        glutReshapeFunc(self.changeSize)

        self.startTime = time.time()
        self.initCamera()
        self.paintGL()
        glutMainLoop()

    def changeSize(self, width, height):
        self.camera.updateWindowSize(width, height)
        self.paintGL()
        glutPostRedisplay()

    def drawLeaf(self):
        glColor3f(0.6, 0.5, 0.4)
        glBegin(GL_TRIANGLES)
        for i in range(len(self.leaf.face)):
            for j in range(3):
                glVertex3f(self.leaf.vertices[self.leaf.face[i][j]][0], self.leaf.vertices[self.leaf.face[i][j]][1],
                           self.leaf.vertices[self.leaf.face[i][j]][2])
        glEnd()
        glFlush()

    def drawEdge(self):
        glColor3f(1, 0, 0)
        glBegin(GL_POINTS)
        for i in range(len(self.leaf.vertices)):
            glVertex3f(self.leaf.vertices[i][0], self.leaf.vertices[i][1],
                       self.leaf.vertices[i][2])
        glEnd()
        glFlush()

    def drawLeafPoints(self):
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_POINTS)
        for i in range(len(self.points)):
            glVertex3f(self.points[i][0], self.points[i][1], self.points[i][2])
        glEnd()
        glFlush()

    def initCamera(self):
        self.camera.eye = np.array([[0, 0, 5]])
        self.camera.at = np.array([[0, 0.5, 0]])
        self.camera.upVec = np.array([[0, 1, 0]])
        self.camera.fov = 45
        self.camera.widgetSize = [glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)]

    def paintGL(self):
        glShadeModel(GL_SMOOTH)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.draw()
        if type == 0:
            glutDisplayFunc(self.drawLeaf)
        elif type == 1:
            glutDisplayFunc(self.drawLeafPoints)
        elif type == 2:
            glutDisplayFunc(self.drawEdge)

    def mouseClick(self, button, state, x, y):
        self.mouseButton = button
        if (button == GLUT_LEFT_BUTTON or button == GLUT_RIGHT_BUTTON) and state == GLUT_DOWN:
            self.startX = x
            self.startY = y
        else:
            if button == GLUT_WHEEL_UP or button == GLUT_WHEEL_DOWN:
                if button == GLUT_WHEEL_UP:
                    self.camera.zoom(120)
                elif button == GLUT_WHEEL_DOWN:
                    self.camera.zoom(-120)
                self.paintGL()
                glutPostRedisplay()

    def mouseMove(self, x, y):
        nDX = x - self.startX
        nDY = y - self.startY
        self.startX = x
        self.startY = y
        nDX /= 10
        nDY /= 10
        if self.mouseButton == GLUT_LEFT_BUTTON:
            self.camera.orbit(nDX, nDY)
        elif self.mouseButton == GLUT_RIGHT_BUTTON:
            self.camera.pan(nDX, nDY)

        self.paintGL()
        glutPostRedisplay()


if __name__ == '__main__':
    leaf = ParametricLeaf()
    points = PcdRead(Pcdfilename)
    ParaReader(paraFilename, leaf)
    leaf.computeParamLeafModel()
    leafRender = Leaf_Render(points, leaf)
