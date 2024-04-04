import sys

import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QTabWidget, QTableWidgetItem, QHeaderView
import design
from PyQt5.QtGui import QPixmap
import cv2
import imutils
import subprocess
import numpy as np

sys.path.insert(1, '../method/')
import method


class MainWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.goButton.clicked.connect(self.goButtonClicked)
        self.helpButton.clicked.connect(self.showHelp)
        self.optimBoundsButton.clicked.connect(self.calculateOptim)
        self.isTest.stateChanged.connect(self.testSetted)
        self.isMain11.stateChanged.connect(self.main11Setted)
        self.isMain12.stateChanged.connect(self.main12Setted)
        self.isMain21.stateChanged.connect(self.main21Setted)
        self.isMain22.stateChanged.connect(self.main22Setted)
        self.elems = [self.isMain11, self.isMain12, self.isMain21, self.isMain22, self.isTest]
        self.splineCoefTable.setColumnCount(7)
        self.splineCoefTable.setHorizontalHeaderLabels(["i", "xi-1", "xi", "ai", "bi", "ci", "di"])
        self.splineCoefTable.verticalHeader().hide()
        self.ErrorRateTable.setColumnCount(11)
        self.ErrorRateTable.setHorizontalHeaderLabels(
            ["i", "xi", "F(xi)", "S(xi)", "F(xi)-S(xi)", "F'(xi)", "S'(xi)", "F'(xi)-S'(xi)", "F''(xi)", "S''(xi)",
             "F''(xi)-S''(xi)"])
        self.ErrorRateTable.verticalHeader().hide()

        self.main1info = """Сетка сплайна: n = %
Контрольная сетка: N = %
Погрешность сплайна на 
контрольной сетке
max|F(xi)-S(xi)| = % 
при x = %
Погрешность производной на 
контрольной сетке
max|F'(xi)-S'(xi)| = % 
при x = %
Погрешность второй производной на 
контрольной сетке
max|F''(xi)-S''(xi)| = %
при x = %"""

    def goButtonClicked(self):
        if self.isMain11.isChecked():
            self.processMain(method.calculateMain11, method.Task1Func1, method.dTask1Func1, method.d2Task1Func1)
        elif self.isMain12.isChecked():
            self.processMain(method.calculateMain12, method.Task1Func2, method.dTask1Func2, method.d2Task1Func2)
        elif self.isMain21.isChecked():
            self.processMain(method.calculateMain21, method.Task2Func1, method.dTask2Func1, method.d2Task2Func1)
        elif self.isMain22.isChecked():
            self.processMain(method.calculateMain22, method.Task2Func2, method.dTask2Func2, method.d2Task2Func2)
        elif self.isTest.isChecked():
            self.processMain(method.calculateTf, method.tf, method.dtf, method.d2tf)

    def drawPlanes(self, data, name):
        plt.figure(figsize=(16, 8))
        for set_ in data:
            plt.plot(set_[0], set_[1])
        plt.legend(('S(x)', 'F(x)', '|F(x)-S(x)|'))
        plt.savefig("../planes/" + name + ".png")
        plt.clf()

    def drawPlanesDense(self, coefs, func, xval, name, legend):
        left = float(self.aBox.text())
        right = float(self.bBox.text())
        X = np.linspace(left, right, 10000)
        plt.figure(figsize=(16, 8))
        spl = [method.splain(*coefs, xval, 0, x) for x in X]
        fun = [func(x) for x in X]
        err = [abs(spl[i]-fun[i]) for i in range(len(spl))]
        plt.plot(X, spl)
        plt.plot(X, fun)
        plt.plot(X, err)
        plt.legend(legend)
        plt.savefig("../planes/" + name + ".png")
        plt.clf()

    def calculateOptim(self):
        if self.isMain11.isChecked():
            self.aBoundBox.setText(str(method.d2Task1Func1(float(self.aBox.text()))))
            self.bBoundBox.setText(str(method.d2Task1Func1(float(self.bBox.text()))))
        elif self.isMain12.isChecked():
            self.aBoundBox.setText(str(method.d2Task1Func2(float(self.aBox.text()))))
            self.bBoundBox.setText(str(method.d2Task1Func2(float(self.bBox.text()))))
        elif self.isMain21.isChecked():
            self.aBoundBox.setText(str(method.d2Task2Func1(float(self.aBox.text()))))
            self.bBoundBox.setText(str(method.d2Task2Func1(float(self.bBox.text()))))
        elif self.isMain22.isChecked():
            self.aBoundBox.setText(str(method.d2Task2Func2(float(self.aBox.text()))))
            self.bBoundBox.setText(str(method.d2Task2Func2(float(self.bBox.text()))))
        elif self.isTest.isChecked():
            self.aBoundBox.setText(str(method.d2tf(float(self.aBox.text()))))
            self.bBoundBox.setText(str(method.d2tf(float(self.bBox.text()))))

    def parseCoefs(self):
        return int(self.nBox.text()), float(self.aBoundBox.text()), float(self.bBoundBox.text()), float(
            self.aBox.text()), float(self.bBox.text())

    def getErr(self, y1, y2):
        return [abs(y1[i] - y2[i]) for i in range(len(y1))]

    def resizeImage(self, imageName, plane):
        height = plane.size().height()
        image = cv2.imread(imageName)
        resized = imutils.resize(image, height=height)
        cv2.imwrite(imageName, resized)
        return QPixmap(imageName)

    def clamp(self, number):
        prec = 9
        index = number.find("e")
        if index == -1:
            return number[:prec]
        return number[0:min(prec, index)] + number[index:]

    def addRowToTable(self, table, data):
        table.insertRow(table.rowCount())
        rowCount = table.rowCount()
        columnCount = table.columnCount()
        for j in range(columnCount):
            table.setItem(rowCount - 1, j, QTableWidgetItem(self.clamp(str(data[j]))))

    def clearTable(self, table):
        rowCount = table.rowCount()
        for i in range(0, rowCount):
            table.removeRow(0)

    def fillSplineTable(self, a, b, c, d, xval):
        self.clearTable(self.splineCoefTable)
        for i in range(len(a)):
            self.addRowToTable(self.splineCoefTable, [i, xval[i], xval[i + 1], a[i], b[i], c[i], d[i]])

    def fillMainTable(self, data, err, err_, err__):
        self.clearTable(self.ErrorRateTable)
        for i in range(len(data[0])):
            self.addRowToTable(self.ErrorRateTable, [i, data[0][i], data[2][1][i], data[2][0][i], err[i],
                                                     data[3][1][i], data[3][0][i], err_[i],
                                                     data[4][1][i], data[4][0][i], err__[i]])

    def fillInfo(self, text, data):
        for val in data:
            text = text.replace("%", str(val), 1)
        return text

    def showHelp(self):
        subprocess.run("python helpui.py")

    def processMain(self, calcf, func, funcd, func2d):
        n, A, B, a_, b_ = self.parseCoefs()
        data = calcf(n, a_, b_, A, B)
        err = self.getErr(*(data[2]))
        err_ = self.getErr(*(data[3]))
        err__ = self.getErr(*(data[4]))
        #self.drawPlanes([[data[0], data[2][0]], [data[0], data[2][1]], [data[0], err]], "func")
        #self.drawPlanes([[data[0], data[3][0]], [data[0], data[3][1]], [data[0], err_]], "funcd")
        #self.drawPlanes([[data[0], data[4][0]], [data[0], data[4][1]], [data[0], err__]], "func2d")
        coefs = data[1]
        xval = data[5]
        self.drawPlanesDense(coefs, func, xval, "func", ('S(x)', 'F(x)', '|F(x)-S(x)|'))
        coefs = method.splainDer(*coefs)
        self.drawPlanesDense(coefs, funcd, xval, "funcd", ('S\'(x)', 'F\'(x)', '|F\'(x)-S\'(x)|'))
        coefs = method.splainDer(*coefs)
        self.drawPlanesDense(coefs, func2d, xval, "func2d", ('S\'\'(x)', 'F\'\'(x)', '|F\'\'(x)-S\'\'(x)|'))
        self.FuncPlane.setPixmap(self.resizeImage("../planes/func.png", self.FuncPlane))
        self.DerPlane.setPixmap(self.resizeImage("../planes/funcd.png", self.DerPlane))
        self.Der2Plane.setPixmap(self.resizeImage("../planes/func2d.png", self.Der2Plane))
        self.fillSplineTable(*(data[1]), data[5])
        self.info.setText(self.fillInfo(self.main1info, [n, 2 * n + 1, max(err), data[0][err.index(max(err))]
            , max(err_), data[0][err_.index(max(err_))], max(err__), data[0][err__.index(max(err__))]]))
        self.fillMainTable(data, err, err_, err__)

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
