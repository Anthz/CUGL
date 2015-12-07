#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QString>
#include <QLabel>
#include <QVBoxLayout>
#include <QOpenGLWidget>
#include <QTableWidget>
#include <QGroupBox>
#include <QTabWidget>
#include <QSplitter>

#include "glwidget.h"
#include "glsettings.h"
#include "cusettings.h"
#include "outputsettings.h"
#include "utilities.h"

namespace Ui
{
	class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget* parent = 0);
	~MainWindow();

	QSize sizeHint() const;
	QString title;

private:
	void InitOpenGLWidget();
	void InitSettings();

	Ui::MainWindow* ui;
	QSplitter* mainSplitter, *settingsSplitter;
	QVBoxLayout *mainLayout, *openglLayout, *cudaLayout;
	GLSettings* openglSettings;
	CUSettings* cudaSettings;
	OutputSettings *outputSettings;
	GLWidget* gl;
	QTabWidget *openglTabs, *cudaTabs;
	QSizePolicy top, bottom;

	private slots:
	void on_actionExit_triggered();
	void on_actionOpenGL_triggered(bool checked);
	void on_actionOpenGL_2_triggered(bool checked);
	void on_actionCUDA_triggered(bool checked);
	void on_actionOutput_triggered(bool checked);
};

#endif // MAINWINDOW_H


