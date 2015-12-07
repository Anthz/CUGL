/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.5.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionExit;
    QAction *actionOpenGL;
    QAction *actionOpenGL_2;
    QAction *actionCUDA;
    QAction *actionOutput;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuView;
    QMenu *menuSettings;
    QMenu *menuHelp;
    QMenu *menuAbout;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1200, 900);
        actionExit = new QAction(MainWindow);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        actionOpenGL = new QAction(MainWindow);
        actionOpenGL->setObjectName(QStringLiteral("actionOpenGL"));
        actionOpenGL->setCheckable(true);
        actionOpenGL->setChecked(true);
        actionOpenGL_2 = new QAction(MainWindow);
        actionOpenGL_2->setObjectName(QStringLiteral("actionOpenGL_2"));
        actionOpenGL_2->setCheckable(true);
        actionOpenGL_2->setChecked(true);
        actionCUDA = new QAction(MainWindow);
        actionCUDA->setObjectName(QStringLiteral("actionCUDA"));
        actionCUDA->setCheckable(true);
        actionCUDA->setChecked(true);
        actionOutput = new QAction(MainWindow);
        actionOutput->setObjectName(QStringLiteral("actionOutput"));
        actionOutput->setCheckable(true);
        actionOutput->setChecked(false);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 400, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuView = new QMenu(menuBar);
        menuView->setObjectName(QStringLiteral("menuView"));
        menuSettings = new QMenu(menuView);
        menuSettings->setObjectName(QStringLiteral("menuSettings"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QStringLiteral("menuHelp"));
        menuAbout = new QMenu(menuBar);
        menuAbout->setObjectName(QStringLiteral("menuAbout"));
        MainWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        statusBar->setEnabled(true);
        statusBar->setFocusPolicy(Qt::NoFocus);
        statusBar->setAutoFillBackground(false);
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuView->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuBar->addAction(menuAbout->menuAction());
        menuFile->addAction(actionExit);
        menuView->addAction(actionOpenGL);
        menuView->addAction(menuSettings->menuAction());
        menuSettings->addAction(actionOpenGL_2);
        menuSettings->addAction(actionCUDA);
        menuSettings->addAction(actionOutput);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        actionExit->setText(QApplication::translate("MainWindow", "Exit", 0));
        actionExit->setShortcut(QApplication::translate("MainWindow", "Esc", 0));
        actionOpenGL->setText(QApplication::translate("MainWindow", "OpenGL", 0));
#ifndef QT_NO_TOOLTIP
        actionOpenGL->setToolTip(QApplication::translate("MainWindow", "OpenGL Window", 0));
#endif // QT_NO_TOOLTIP
        actionOpenGL->setShortcut(QApplication::translate("MainWindow", "Shift+O", 0));
        actionOpenGL_2->setText(QApplication::translate("MainWindow", "OpenGL", 0));
#ifndef QT_NO_TOOLTIP
        actionOpenGL_2->setToolTip(QApplication::translate("MainWindow", "OpenGL Settings", 0));
#endif // QT_NO_TOOLTIP
        actionOpenGL_2->setShortcut(QApplication::translate("MainWindow", "Shift+Q", 0));
        actionCUDA->setText(QApplication::translate("MainWindow", "CUDA", 0));
#ifndef QT_NO_TOOLTIP
        actionCUDA->setToolTip(QApplication::translate("MainWindow", "CUDA Settings", 0));
#endif // QT_NO_TOOLTIP
        actionCUDA->setShortcut(QApplication::translate("MainWindow", "Shift+W", 0));
        actionOutput->setText(QApplication::translate("MainWindow", "Output", 0));
#ifndef QT_NO_TOOLTIP
        actionOutput->setToolTip(QApplication::translate("MainWindow", "Output Settings", 0));
#endif // QT_NO_TOOLTIP
        actionOutput->setShortcut(QApplication::translate("MainWindow", "Shift+E", 0));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", 0));
        menuView->setTitle(QApplication::translate("MainWindow", "View", 0));
        menuSettings->setTitle(QApplication::translate("MainWindow", "Settings", 0));
        menuHelp->setTitle(QApplication::translate("MainWindow", "Help", 0));
        menuAbout->setTitle(QApplication::translate("MainWindow", "About", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
