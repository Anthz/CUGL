#include "glwidget.h"
#include "glsettings.h"
#include "cusettings.h"

void CUExecuteKernel(std::vector<void*> *params);	//std::vector<void*> *params

namespace
{
	GLWidget* glPtr = nullptr;
}

std::vector<Shader*> GLWidget::ShaderList;

GLWidget::GLWidget(QWidget* parent) : QOpenGLWidget(parent)
{
	glPtr = this;
	parentWidget = parent;
	projMatrix = new QMatrix4x4();
	viewMatrix = new QMatrix4x4();
	viewMatrix->lookAt(
		QVector3D(0.0, 0.0, 300.0), // Eye
		QVector3D(0.0, 0.0, 0.0), // Focal Point
		QVector3D(0.0, 1.0, 0.0)); // Up vector
	drawMode = GL_TRIANGLES;
	play = false;
	timer.start();
	setFocusPolicy(Qt::StrongFocus);
}

GLWidget::~GLWidget()
{
	for each (Shader *s in ShaderList)
	{
		delete s;
	}

	//delete glFuncs;
}

void GLWidget::initializeGL()
{
	initializeOpenGLFunctions();

	glFuncs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
	if(!glFuncs)
	{
		qWarning() << "Could not obtain required OpenGL context version";
		exit(1);
	}

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	//glEnable(GL_CULL_FACE);

	Shader *shader = new Shader("Textured Particle Shader", "./Shaders/t_bb_Particle.vert", "./Shaders/t_bb_Particle.frag");
	ShaderList.push_back(shader);

	Shader *shader2 = new Shader("Particle Shader", "./Shaders/bb_Particle.vert", "./Shaders/bb_Particle.frag");
	ShaderList.push_back(shader2);

	int r, g, b, a;
	QColor c("#6495ED");
	c.getRgb(&r, &g, &b, &a);
	glClearColor(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
}

//context made current in init/paint/resizeGL
void GLWidget::paintGL()
{
	UpdateFPS(); //fps broken due to single thread

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//map buffers
	//call kernel
	/*for(int i = 0; i < ShaderList.size(); ++i)
	{
	ShaderList.at(i)->SetUniform("uProjMatrix", *projMatrix);
	ShaderList.at(i)->SetUniform("uViewMatrix", *viewMatrix);
	ShaderList.at(i)->SetUniform("uInvViewMatrix", viewMatrix->inverted()); //returns inverse matrix (camera position)
	}*/

	if(play)
	{
		if(GLSettings::ObjectList.size() > 0)
		{
			std::vector<void*> params;

			for(int i = 0; i < GLSettings::BufferList.size(); ++i)
			{
				if(GLSettings::BufferList.at(i)->Cuda())	//use CUSettings list instead?
				{
					void *devPtr = CUGLBuffer::MapResource(GLSettings::BufferList.at(i)->CudaBuf());
					params.push_back(devPtr);
				}
			}

			CUExecuteKernel(&params);	//, GLSettings::ObjectList.at(0)->instances //hardcoded for 1 object

			for(int i = 0; i < GLSettings::BufferList.size(); ++i)
			{
				if(GLSettings::BufferList.at(i)->Cuda())
				{
					CUGLBuffer::UnmapResource(GLSettings::BufferList.at(i)->CudaBuf());
				}
			}

			for each (Object *o in GLSettings::ObjectList)
			{
				o->Draw(drawMode, false);
				update();
			}
		}
	}
}

int GLWidget::Width()
{
	return glPtr->width;
}

int GLWidget::Height()
{
	return glPtr->height;
}

void GLWidget::DoneCurrent()
{
	glPtr->doneCurrent();
}

void GLWidget::MakeCurrent()
{
	glPtr->makeCurrent();
}

void GLWidget::Resize(int w, int h)
{
	glPtr->resizeGL(w, h);
}

void GLWidget::FOV(int i)
{
}

void GLWidget::DrawMode(int i)
{
	switch(i)
	{
	case 0:
		glPtr->drawMode = GL_TRIANGLES;
		break;
	case 1:
		glPtr->drawMode = GL_TRIANGLE_STRIP;
		break;
	case 2:
		glPtr->drawMode = GL_POINTS;
		break;
	}
}

void GLWidget::VSync(bool b)
{
}

void GLWidget::MSAA(bool b)
{
}

void GLWidget::Play(bool b)
{
	glPtr->play = b;
}

QMatrix4x4 *GLWidget::ProjMatrix()
{
	return glPtr->projMatrix;
}

QMatrix4x4 *GLWidget::ViewMatrix()
{
	return glPtr->viewMatrix;
}

void GLWidget::keyPressEvent(QKeyEvent* e)
{
	static float xRot = 0.0f;
	static float yRot = 0.0f;
	static float zRot = 0.0f;

	if(e->key() == Qt::Key_O)
	{
		makeCurrent();
		//yRot += 3.0f;
		viewMatrix->rotate(3.0f, QVector3D(0.0, 1.0, 0.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_P)
	{
		makeCurrent();
		//yRot -= 3.0f;
		viewMatrix->rotate(-3.0f, QVector3D(0.0, 1.0, 0.0));
		update();
		doneCurrent();
	}

	if(e->key() == Qt::Key_Q)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(0.0, 3.0, 0.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_W)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(0.0, 0.0, 3.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_E)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(0.0, -3.0, 0.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_A)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(-3.0, 0.0, 0.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_S)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(0.0, 0.0, -3.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_D)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(3.0, 0.0, 0.0));
		update();
		doneCurrent();
	}
}

void GLWidget::resizeGL(int w, int h)
{
	width = w;
	height = h;

	/*int side = qMin(width, height);
	glViewport((width - side) / 2, (height - side) / 2, side, side);*/

	const qreal retinaScale = devicePixelRatio();
	glViewport(0, 0, width * retinaScale, height * retinaScale);

	projMatrix->setToIdentity();
	//projMatrix->ortho(-0.5f, 0.5f, -0.5f, 0.5f, 0.1f, 1000.0f);
	projMatrix->perspective(45.0f, width / float(height), 0.01f, 10000.0f);
}

void GLWidget::UpdateFPS()
{
	static QString title = parentWidget->windowTitle();
	static int frameCount;
	static int elapsedTime = 0;
	elapsedTime += timer.restart();

	if(elapsedTime >= 250) //update every 1/4 second
	{
		int fps = ((double)frameCount / ((double)elapsedTime / 1000.0)) + 0.5;
		QString s = title + " | FPS: " + QString::number(fps);
		parentWidget->setWindowTitle(s);
		//glfwSetWindowTitle(mainWND->Handle(), s.c_str());
		//std::string str = "Frames: " + std::to_string(frameCount) + " Elapsed: " + std::to_string(elapsedTime) + " FPS: " + std::to_string(fps) + "\n";
		//printf(str.c_str());
		frameCount = 0;
		elapsedTime = 0;
	}

	frameCount++;
}

QSize GLWidget::minimumSizeHint() const
{
	return QSize(200, 200);
}

QSize GLWidget::sizeHint() const
{
	return QSize(600, 600);
}

