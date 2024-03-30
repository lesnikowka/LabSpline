import sys

import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QTabWidget, QTableWidgetItem, QHeaderView
import design
from PyQt5.QtGui import QPixmap
import cv2
import imutils
import subprocess

import method.method


#from method import method

class MainWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.goButton.clicked.connect(self.goButtonClicked)
        self.isTest.stateChanged.connect(self.testSetted)
        self.isMain11.stateChanged.connect(self.main11Setted)
        self.isMain12.stateChanged.connect(self.main12Setted)
        self.isMain21.stateChanged.connect(self.main21Setted)
        self.isMain22.stateChanged.connect(self.main22Setted)
        self.elems = [self.isMain11, self.isMain12, self.isMain21, self.isMain22, self.isTest]

    def goButtonClicked(self):
        if self.isMain11.isChecked():
            self.processMain1(method.method.calculateMain11)
        elif self.isMain12.isChecked():
            self.processMain1(method.method.calculateMain12)

    def drawPlanes(self, data, name):
        for set_ in data:
            plt.plot(set_[0], set_[1])
        plt.savefig("../planes/" + name + ".png")
        plt.clf()

    def parseCoefs(self):
        return int(self.nBox.text()), float(self.aBoundBox.text()), float(self.bBoundBox.text()), float(self.aBox.text()), float(self.bBox.text())
    def getErr(self, y1, y2):
        return [abs(y1[i] - y2[i]) for i in range(len(y1))]
    def resizeImage(self, imageName, plane):
        height = plane.size().height()
        image = cv2.imread(imageName)
        resized = imutils.resize(image, height=height)
        cv2.imwrite(imageName, resized)
        return  QPixmap(imageName)

    def processMain1(self, calcf):
        n, A, B, a_, b_ = self.parseCoefs()
        data = calcf(n, a_, b_, A, B)
        err = self.getErr(*(data[2]))
        err_ = self.getErr(*(data[3]))
        err__ = self.getErr(*(data[4]))
        self.drawPlanes([[data[0], data[2][0]], [data[0], data[2][1]], [data[0], err]], "func")
        self.drawPlanes([[data[0], data[3][0]], [data[0], data[3][1]], [data[0], err_]], "funcd")
        self.drawPlanes([[data[0], data[4][0]], [data[0], data[4][1]], [data[0], err__]], "func2d")
        self.FuncPlane.setPixmap(self.resizeImage("../planes/func.png", self.FuncPlane))
        self.DerPlane.setPixmap(self.resizeImage("../planes/funcd.png", self.DerPlane))
        self.Der2Plane.setPixmap(self.resizeImage("../planes/func2d.png", self.Der2Plane))





    def setUnchecked(self, ignored):
        for i in range(len(self.elems)):
            if not self.elems[i] is ignored:
                self.elems[i].setChecked(False)

    def testSetted(self):
        self.setUnchecked(self.isTest)
    def main11Setted(self):
        self.setUnchecked(self.isMain11)
    def main12Setted(self):
        self.setUnchecked(self.isMain12)
    def main21Setted(self):
        self.setUnchecked(self.isMain21)
    def main22Setted(self):
        self.setUnchecked(self.isMain22)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()