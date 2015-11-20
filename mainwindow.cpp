#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget* parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	mainLayout = new QVBoxLayout();
	mainSplitter = new QSplitter();

	top = QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
	top.setVerticalStretch(2.5);
	bottom = QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
	bottom.setVerticalStretch(1);

	InitOpenGLWidget();
	InitSettings();

	//mainLayout->setSizeConstraint(QLayout::SetFixedSize);

	centralWidget()->setLayout(mainLayout);

	title = "CUGL - CUDA/OpenGL Framework";
	setWindowTitle(title);
	ui->statusBar->showMessage("Testing...", 5000);
}

MainWindow::~MainWindow()
{
	/*if(openglTabs)
		delete openglTabs;
	if(cudaTabs)
		delete cudaTabs;
	if(cudaSettings)
		delete cudaSettings;
	if(openglSettings)
		delete openglSettings;
	if(openglLayout)
		delete openglLayout;
	if(cudaLayout)
		delete cudaLayout;
	if(gl)
		delete gl;
	if(settings)
		delete settings;
	if(mainSplitter)
		delete mainSplitter;
	if(mainLayout)
		delete mainLayout;
	if(ui)
		delete ui;*/
}

QSize MainWindow::sizeHint() const
{
	return QSize(800, 800);
}

void MainWindow::InitOpenGLWidget()
{
	gl = new GLWidget(this);
	gl->setSizePolicy(top);
	mainLayout->addWidget(gl);
}

void MainWindow::InitSettings()
{
	settings = new QWidget();
	settingsLayout = new QHBoxLayout();

	openglSettings = new GLSettings();

	cudaSettings = new CUSettings();

	mainSplitter->addWidget(openglSettings);
	mainSplitter->addWidget(cudaSettings);
	mainSplitter->setSizePolicy(bottom);

	//settingsLayout->addWidget(openglSettings);
	//settingsLayout->addWidget(cudaSettings);

	//settings->setLayout(settingsLayout);
	//settings->setSizePolicy(bottom);

	mainLayout->addWidget(mainSplitter);
}

void MainWindow::on_actionExit_triggered()
{
	QCoreApplication::quit();
}

void MainWindow::on_actionOpenGL_triggered(bool checked)
{
	gl->setHidden(!checked);
}

void MainWindow::on_actionOpenGL_2_triggered(bool checked)
{
	openglSettings->setHidden(!checked);
}

void MainWindow::on_actionCUDA_triggered(bool checked)
{
	cudaSettings->setHidden(!checked);
}

