# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1247, 791)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(360, 150, 811, 461))
        self.tabWidget.setObjectName("tabWidget")
        self.FuncPlaneTab = QtWidgets.QWidget()
        self.FuncPlaneTab.setObjectName("FuncPlaneTab")
        self.FuncPlane = QtWidgets.QLabel(self.FuncPlaneTab)
        self.FuncPlane.setGeometry(QtCore.QRect(110, 60, 511, 301))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FuncPlane.sizePolicy().hasHeightForWidth())
        self.FuncPlane.setSizePolicy(sizePolicy)
        self.FuncPlane.setText("")
        self.FuncPlane.setObjectName("FuncPlane")
        self.tabWidget.addTab(self.FuncPlaneTab, "")
        self.DerPlaneTab = QtWidgets.QWidget()
        self.DerPlaneTab.setObjectName("DerPlaneTab")
        self.DerPlane = QtWidgets.QLabel(self.DerPlaneTab)
        self.DerPlane.setGeometry(QtCore.QRect(110, 50, 511, 301))
        self.DerPlane.setText("")
        self.DerPlane.setObjectName("DerPlane")
        self.tabWidget.addTab(self.DerPlaneTab, "")
        self.Der2PlaneTab = QtWidgets.QWidget()
        self.Der2PlaneTab.setObjectName("Der2PlaneTab")
        self.Der2Plane = QtWidgets.QLabel(self.Der2PlaneTab)
        self.Der2Plane.setGeometry(QtCore.QRect(160, 80, 511, 301))
        self.Der2Plane.setText("")
        self.Der2Plane.setObjectName("Der2Plane")
        self.tabWidget.addTab(self.Der2PlaneTab, "")
        self.ErrorRateTab = QtWidgets.QWidget()
        self.ErrorRateTab.setObjectName("ErrorRateTab")
        self.ErrorRateTable = QtWidgets.QTableWidget(self.ErrorRateTab)
        self.ErrorRateTable.setGeometry(QtCore.QRect(60, 100, 581, 251))
        self.ErrorRateTable.setObjectName("ErrorRateTable")
        self.ErrorRateTable.setColumnCount(0)
        self.ErrorRateTable.setRowCount(0)
        self.tabWidget.addTab(self.ErrorRateTab, "")
        self.GridValuesTab = QtWidgets.QWidget()
        self.GridValuesTab.setObjectName("GridValuesTab")
        self.GridValuesTable = QtWidgets.QTableWidget(self.GridValuesTab)
        self.GridValuesTable.setGeometry(QtCore.QRect(30, 20, 611, 311))
        self.GridValuesTable.setObjectName("GridValuesTable")
        self.GridValuesTable.setColumnCount(0)
        self.GridValuesTable.setRowCount(0)
        self.tabWidget.addTab(self.GridValuesTab, "")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 11, 231, 601))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.isMain1 = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.isMain1.setChecked(True)
        self.isMain1.setObjectName("isMain1")
        self.verticalLayout.addWidget(self.isMain1)
        self.isMain2 = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.isMain2.setObjectName("isMain2")
        self.verticalLayout.addWidget(self.isMain2)
        self.isTest = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.isTest.setObjectName("isTest")
        self.verticalLayout.addWidget(self.isTest)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 1, 1, 1)
        self.bBoundBox = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.bBoundBox.setObjectName("bBoundBox")
        self.gridLayout.addWidget(self.bBoundBox, 3, 2, 1, 1)
        self.bBox = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.bBox.setObjectName("bBox")
        self.gridLayout.addWidget(self.bBox, 1, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.aBox = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.aBox.setObjectName("aBox")
        self.gridLayout.addWidget(self.aBox, 0, 2, 1, 1)
        self.nBox = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.nBox.setObjectName("nBox")
        self.gridLayout.addWidget(self.nBox, 4, 2, 1, 1)
        self.aBoundBox = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.aBoundBox.setObjectName("aBoundBox")
        self.gridLayout.addWidget(self.aBoundBox, 2, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.goButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.goButton.setObjectName("goButton")
        self.verticalLayout.addWidget(self.goButton)
        self.label_9 = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setObjectName("label_9")
        self.verticalLayout.addWidget(self.label_9)
        self.info = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.info.setText("")
        self.info.setObjectName("info")
        self.verticalLayout.addWidget(self.info)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1247, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Spline"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FuncPlaneTab), _translate("MainWindow", "Гр. функций"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.DerPlaneTab), _translate("MainWindow", "Гр. производных"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Der2PlaneTab), _translate("MainWindow", "Гр. 2 производных"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ErrorRateTab), _translate("MainWindow", "Таблица погрешностей"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.GridValuesTab), _translate("MainWindow", "Таблица значений в узлах"))
        self.label_5.setText(_translate("MainWindow", "Тип задачи: "))
        self.isMain1.setText(_translate("MainWindow", "Основная 1"))
        self.isMain2.setText(_translate("MainWindow", "Основная 2"))
        self.isTest.setText(_translate("MainWindow", "Тестовая"))
        self.label_2.setText(_translate("MainWindow", "b = "))
        self.label_4.setText(_translate("MainWindow", "S\'\'(b) = "))
        self.bBoundBox.setText(_translate("MainWindow", "0"))
        self.bBox.setText(_translate("MainWindow", "1"))
        self.label_3.setText(_translate("MainWindow", "S\'\'(a) = "))
        self.label_6.setText(_translate("MainWindow", "Число разбиений:"))
        self.label.setText(_translate("MainWindow", "a = "))
        self.aBox.setText(_translate("MainWindow", "0"))
        self.nBox.setText(_translate("MainWindow", "10"))
        self.aBoundBox.setText(_translate("MainWindow", "0"))
        self.goButton.setText(_translate("MainWindow", "Аппроксимировать"))
        self.label_9.setText(_translate("MainWindow", "Справка:"))