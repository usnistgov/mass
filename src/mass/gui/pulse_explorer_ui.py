# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pulse_picker_ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(923, 558)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.plot1_container = QtWidgets.QWidget(Form)
        self.plot1_container.setObjectName("plot1_container")
        self.horizontalLayout.addWidget(self.plot1_container)
        self.plot2_container = QtWidgets.QWidget(Form)
        self.plot2_container.setObjectName("plot2_container")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.plot2_container)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.plot2 = QtWidgets.QWidget(self.plot2_container)
        self.plot2.setObjectName("plot2")
        self.verticalLayout_3.addWidget(self.plot2)
        self.plot2_controls = QtWidgets.QWidget(self.plot2_container)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot2_controls.sizePolicy().hasHeightForWidth())
        self.plot2_controls.setSizePolicy(sizePolicy)
        self.plot2_controls.setObjectName("plot2_controls")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.plot2_controls)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.le_pulse_index = QtWidgets.QLineEdit(self.plot2_controls)
        self.le_pulse_index.setObjectName("le_pulse_index")
        self.gridLayout_2.addWidget(self.le_pulse_index, 0, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.plot2_controls)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.plot2_controls)
        self.horizontalLayout.addWidget(self.plot2_container)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_4.setText(_translate("Form", "Pulse Indices:"))

