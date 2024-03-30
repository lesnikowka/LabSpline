import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QTabWidget, QTableWidgetItem, QHeaderView
import design
from PyQt5.QtGui import QPixmap
import cv2
import imutils
import subprocess
#from method import method

class MainWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.goButton.clicked.connect(self.goButtonClicked)
        self.isTest.stateChanged.connect(self.testSetted)
        self.isMain1.stateChanged.connect(self.main1Setted)
        self.isMain2.stateChanged.connect(self.main2Setted)

    def goButtonClicked(self):
        pass

    def testSetted(self):
        self.isMain1.setChecked(False)
        self.isMain2.setChecked(False)
    def main1Setted(self):
        self.isTest.setChecked(False)
        self.isMain2.setChecked(False)

    def main2Setted(self):
        self.isMain1.setChecked(False)
        self.isTest.setChecked(False)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()