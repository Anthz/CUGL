#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget* parent) :
QMainWindow(parent),
ui(new Ui::MainWindow)
{
	Logger::InitLogger();
	ui->setupUi(this);

	mainLayout = new QVBoxLayout();
	mainSplitter = new QSplitter(Qt::Orientation::Vertical);
	settingsSplitter = new QSplitter();

	top = QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
	top.setVerticalStretch(3);
	bottom = QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
	bottom.setVerticalStretch(1);
	bottom.setHorizontalStretch(1);

	InitOpenGLWidget();
	InitSettings();

	mainLayout->addWidget(mainSplitter);

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
	return QSize(1200, 900);
}

void MainWindow::InitOpenGLWidget()
{
	gl = new GLWidget(this);
	gl->setSizePolicy(top);
	mainSplitter->addWidget(gl);
}

void MainWindow::InitSettings()
{
	//fix weighting
	openglSettings = new GLSettings(this);
	openglSettings->setSizePolicy(bottom);
	cudaSettings = new CUSettings(this);
	cudaSettings->setSizePolicy(bottom);
	outputSettings = new OutputSettings(this);
	outputSettings->setSizePolicy(bottom);

	settingsSplitter->addWidget(openglSettings);
	settingsSplitter->addWidget(cudaSettings);
	settingsSplitter->addWidget(outputSettings);
	settingsSplitter->setSizePolicy(bottom);

	mainSplitter->addWidget(settingsSplitter);
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

void MainWindow::on_actionOutput_triggered(bool checked)
{
	outputSettings->setHidden(!checked);
}

