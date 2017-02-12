#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "cuda_profiler_api.h"
#include <QTextStream>

MainWindow::MainWindow(QWidget* parent) :
QMainWindow(parent),
ui(new Ui::MainWindow)
{
	Logger::InitLogger();
	ui->setupUi(this);

	mainLayout = new QVBoxLayout();
	mainSplitter = new QSplitter(Qt::Orientation::Vertical);
	bottomSplitter = new QSplitter();
	sideSplitter = new QSplitter();

	top = QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
	top.setVerticalStretch(3);
	top.setHorizontalStretch(3);
	bottom = QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
	bottom.setVerticalStretch(1);
	bottom.setHorizontalStretch(1);

	InitOpenGLWidget();
	InitSettings();

	//output sidebar
	outputSettings = new OutputSettings(this);
	outputSettings->setSizePolicy(bottom);
	mainSplitter->setSizePolicy(top);

	sideSplitter->addWidget(mainSplitter);
	sideSplitter->addWidget(outputSettings);

	mainLayout->addWidget(sideSplitter);

	centralWidget()->setLayout(mainLayout);

	title = "CUGL - CUDA/OpenGL Framework";
	setWindowTitle(title);
	ui->statusBar->showMessage("Testing...", 5000);
}

MainWindow::~MainWindow()
{
 	delete mainLayout;
}

QSize MainWindow::sizeHint() const
{
	return QSize(1200, 1200);
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

	bottomSplitter->addWidget(openglSettings);
	bottomSplitter->addWidget(cudaSettings);
	bottomSplitter->setSizePolicy(bottom);

	mainSplitter->addWidget(bottomSplitter);
}

void MainWindow::on_actionExit_triggered()
{
	cudaProfilerStop();
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

void MainWindow::on_actionSave_Project_triggered()
{
	//save seed
	//loop through tex, then buffers, then objects
	//make pure virtual and pass in stream
	QString filePath = QFileDialog::getSaveFileName(this, "Save Project File", QDir::currentPath(), "Project Files (*.cuglproj)");
	QFile file(filePath);

	if(file.open(QIODevice::WriteOnly))
	{
		QTextStream outStream(&file);

		for each (Texture *t in GLSettings::TextureList)
		{
			std::vector<QString> varList;	//test perf vs single vector + clear
			t->Save(&outStream, &varList);
		}

		for each (CUGLBuffer *b in GLSettings::BufferList)
		{
			std::vector<QString> varList;
			b->Save(&outStream, &varList);
		}

		for each (Object *o in GLSettings::ObjectList)
		{
			std::vector<QString> varList;
			o->Save(&outStream, &varList);
		}

		file.close();
	}
}

void MainWindow::on_actionLoad_Project_triggered()
{
	QString filePath = QFileDialog::getOpenFileName(this, "Open Project File", QDir::currentPath(), "Project Files (*.cuglproj)");
	QFile file(filePath);

	if(file.open(QIODevice::ReadOnly))
	{
		QTextStream inStream(&file);
		
		QString line;
		QStringList varList;
		std::vector<Texture*> texList;
		std::vector<CUGLBuffer*> bufList;
		std::vector<Object*> objList;

		//index from 1 due to class id (t_/b_/o_)
		while(inStream.readLineInto(&line))
		{
			if(line.startsWith("t_"))
			{
				Texture *t;

				varList = line.split('|');

				t = new Texture(varList.at(1), varList.at(2), varList.at(3).toInt(), varList.at(4).toInt(), (QImage::Format)varList.at(5).toInt(), varList.at(6), varList.at(7), varList.at(8), (bool)varList.at(9).toInt(), varList.at(10).toInt());
				texList.push_back(t);
				openglSettings->AddTexture(t);
			}
			else if(line.startsWith("b_"))
			{
				CUGLBuffer *b;

				varList = line.split('|');

				b = new CUGLBuffer(varList.at(1), varList.at(2).toInt(), varList.at(3), varList.at(4), varList.at(5), varList.at(6), varList.at(7), varList.at(8).toInt(), varList.at(9), (bool)varList.at(10).toInt(), (bool)varList.at(11).toInt());
				bufList.push_back(b);
				openglSettings->AddBuffer(b);
			}
			else if(line.startsWith("o_"))
			{
				Object *o;

				varList = line.split('|');

				std::vector<int> bufIDs;	//split third param into multiple ids (for multi-buffers)
				for each (QString s in varList.at(3).split('~'))
				{
					bufIDs.push_back(s.toInt());
				}

				o = new Object(varList.at(1), varList.at(2).toInt(), &bufIDs, varList.at(4).toInt(), varList.at(5).toInt());
				objList.push_back(o);
				openglSettings->AddObject(o);
			}
		}

		file.close();
	}
}

void MainWindow::on_actionUser_Guide_triggered()
{

}

void MainWindow::on_actionAbout_triggered()
{
	QMessageBox::information(this, "About", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
Mauris auctor mi nunc. Integer ut dapibus tortor. Nullam consectetur nec dui pulvinar gravida. \
Sed dictum enim tortor, eu convallis dui imperdiet ut. Pellentesque habitant morbi tristique senectus \
et netus et malesuada fames ac turpis egestas. Etiam congue vulputate est id fringilla. \
Cras at cursus nisi, quis molestie ex. In sit amet suscipit risus. Quisque auctor risus lectus, \
at consectetur dui blandit et.", QDialogButtonBox::Ok);
}
