#include "glwidget.h"
#include "glsettings.h"
#include "cusettings.h"

void CUSetup();
void CUExecuteKernel(std::vector<void*> *params);	//std::vector<void*> *params

namespace
{
	GLWidget* glPtr = nullptr;
}

std::vector<Shader*> GLWidget::ShaderList;
std::vector<GLuint> GLWidget::FBOList;

GLWidget::GLWidget(QWidget* parent) : QOpenGLWidget(parent)
{
	glPtr = this;
	parentWidget = parent;
	projMatrix = new QMatrix4x4();
	viewMatrix = new QMatrix4x4();
	viewMatrix->lookAt(
		QVector3D(0.0, 0.0, 10.0), // Eye
		QVector3D(0.0, 0.0, 0.0), // Focal Point
		QVector3D(0.0, 1.0, 0.0)); // Up vector
	drawMode = GL_TRIANGLES;
	play = false;
	step = false;
	paramWarning = false;
	srand(time(NULL));
	timer.start();
	setFocusPolicy(Qt::StrongFocus);
}

GLWidget::~GLWidget()
{
	for each (Shader *s in ShaderList)
	{
		delete s;
	}
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

	width = QWidget::width();
	height = QWidget::height();

	glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LESS);

	glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glBlendEquation(GL_ADD);

	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	//configure working directory correctly (./ in VS, ../ otherwise)
	//Shader *shader = new Shader("Textured Particle Shader", "./Shaders/t_bb_Particle_3_3.vert", "./Shaders/t_bb_Particle_3_3.frag");
	Shader *shader = new Shader("Textured Particle Shader", "./Shaders/t_bb_Particle_3_3.vert", "./Shaders/t_bb_Particle_3_3.frag");
	ShaderList.push_back(shader);

	Shader *shader2 = new Shader("Particle Shader", "./Shaders/bb_Particle_3_3.vert", "./Shaders/bb_Particle_3_3.frag");
	ShaderList.push_back(shader2);

	Shader *shader3 = new Shader("Simple Shader", "./Shaders/simple_3_3.vert", "./Shaders/simple_3_3.frag");
	ShaderList.push_back(shader3);

	Shader *shader4 = new Shader("Heightmap Shader", "./Shaders/heightmap.vert", "./Shaders/heightmap.frag");
	ShaderList.push_back(shader4);

	Shader *shader5 = new Shader("FBO Shader", "./Shaders/fbo_3_3.vert", "./Shaders/fbo_3_3.frag");
	ShaderList.push_back(shader5);

	Shader *shader6 = new Shader("Top Probability Shader", "./Shaders/top_prob_3_3.vert", "./Shaders/prob_3_3.frag");
	ShaderList.push_back(shader6);

	Shader *shader7 = new Shader("Bottom Probability Shader", "./Shaders/bottom_prob_3_3.vert", "./Shaders/prob_3_3.frag");
	ShaderList.push_back(shader7);

	int r, g, b, a;
	QColor c("#6495ED");
	c.getRgb(&r, &g, &b, &a);
	glClearColor(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);

	CUSetup();	//setup rand
}

//context made current in init/paint/resizeGL
void GLWidget::paintGL()
{
	UpdateFPS(); //fps broken due to single thread

	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//render to FBO
	//update
	//render FBO tex to screen
	//render other objs to screen
	if(GLSettings::ObjectList.size() > 0)
	{
		if(play || step)
		{
			if(FBOList.size() > 0)
			{
				glBindFramebuffer(GL_FRAMEBUFFER, FBOList.at(0));	//hardcoded for 1 FBO
				glViewport(0, 0, width, height);

				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				for each (Object *o in GLSettings::ObjectList)
				{
					if(o->FBO() == FBOList.at(0))
					{
						o->Draw(drawMode, false);
					}
				}

				glBindFramebuffer(GL_FRAMEBUFFER, 0);
				glViewport(0, 0, width, height);
			}

			std::vector<void*> params;

			for(int i = 0; i < GLSettings::BufferList.size(); ++i)
			{
				if(GLSettings::BufferList.at(i)->Cuda())	//use CUSettings list instead?
				{
					void *devPtr = CUGLBuffer::MapResource(GLSettings::BufferList.at(i)->CudaBuf());
					params.push_back(devPtr);
				}
			}

			if(params.size() > 0)
				CUExecuteKernel(&params);	//, GLSettings::ObjectList.at(0)->instances //hardcoded for 1 object

			for(int i = 0; i < GLSettings::BufferList.size(); ++i)
			{
				if(GLSettings::BufferList.at(i)->Cuda())
				{
					CUGLBuffer::UnmapResource(GLSettings::BufferList.at(i)->CudaBuf());
				}
			}
			step = false;
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for each (Object *o in GLSettings::ObjectList)
		{
			o->Draw(drawMode, false);
		}

		update();
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

void GLWidget::StepForward()
{
	glPtr->step = true;
}

void GLWidget::StepBackward()
{
	//implement step back when saving/reversibility is enabled
	//glPtr->play = b;
}

QMatrix4x4 *GLWidget::ProjMatrix()
{
	return glPtr->projMatrix;
}

QMatrix4x4 *GLWidget::ViewMatrix()
{
	return glPtr->viewMatrix;
}

void GLWidget::CheckFBOStatus()
{
	GLenum status = glPtr->glCheckFramebufferStatus(GL_FRAMEBUFFER);

	const char *err_str = 0;
	char buf[80];

	if(status != GL_FRAMEBUFFER_COMPLETE)
	{
		switch(status)
		{
		case GL_FRAMEBUFFER_UNSUPPORTED:
			err_str = "UNSUPPORTED";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			err_str = "INCOMPLETE ATTACHMENT";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			err_str = "INCOMPLETE DRAW BUFFER";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			err_str = "INCOMPLETE READ BUFFER";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			err_str = "INCOMPLETE MISSING ATTACHMENT";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			err_str = "INCOMPLETE MULTISAMPLE";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			err_str = "INCOMPLETE LAYER TARGETS";
			break;

			// Removed in version #117 of the EXT_framebuffer_object spec
			//case GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT:

		default:
			sprintf(buf, "0x%x", status);
			err_str = buf;
			break;
		}

		Logger::Log("ERROR: glCheckFramebufferStatus() returned " + std::string(err_str));
	}
}

int GLWidget::SetFBOTexture(GLuint id)
{
	//add ability to customise + add more fbos
	GLuint fbo;
	GLuint rbo;
	glPtr->glGenFramebuffers(1, &fbo);
	glPtr->glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glPtr->glGenRenderbuffers(1, &rbo);
	glPtr->glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glPtr->glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, Width(), Height());
	glPtr->glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
	glPtr->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, id, 0);

	//set the list of draw buffers.
	GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
	glPtr->glFuncs->glDrawBuffers(1, drawBuffers);

	CheckFBOStatus();

	glPtr->FBOList.push_back(fbo);

	glPtr->glClearColor(0.0, 0.0, 0.0, 1.0);
	glPtr->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//first clear to ensure drawing

	glPtr->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return fbo;
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
		viewMatrix->rotate(2.0f, QVector3D(0.0, 1.0, 0.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_P)
	{
		makeCurrent();
		//yRot -= 3.0f;
		viewMatrix->rotate(-2.0f, QVector3D(0.0, 1.0, 0.0));
		update();
		doneCurrent();
	}

	if(e->key() == Qt::Key_Q)
	{
		makeCurrent();
		//GLSettings::ObjectList.at(0)->Move(QVector3D(0.0, 3.0, 0.0));
		viewMatrix->translate(QVector3D(0.0, 2.0, 0.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_W)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(0.0, 0.0, 2.0));
		//GLSettings::ObjectList.at(0)->Move(QVector3D(0.0, 0.0, 3.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_E)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(0.0, -2.0, 0.0));
		//GLSettings::ObjectList.at(0)->Move(QVector3D(0.0, -3.0, 0.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_A)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(-2.0, 0.0, 0.0));
		//GLSettings::ObjectList.at(0)->Move(QVector3D(-3.0, 0.0, 0.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_S)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(0.0, 0.0, -2.0));
		//GLSettings::ObjectList.at(0)->Move(QVector3D(0.0, 0.0, -3.0));
		update();
		doneCurrent();
	}
	if(e->key() == Qt::Key_D)
	{
		makeCurrent();
		viewMatrix->translate(QVector3D(2.0, 0.0, 0.0));
		//GLSettings::ObjectList.at(0)->Move(QVector3D(3.0, 0.0, 0.0));
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
	//glViewport(0, 0, width * retinaScale, height * retinaScale);
	glViewport(0, 0, width, height);
	float ratio = (float)width / (float)height;

	projMatrix->setToIdentity();
	//projMatrix->ortho(-25e10 * ratio, 25e10 * ratio, -25e10, 25e10, 0.1f, 10000.0f);
	//projMatrix->ortho(-0.5f, 0.5f, -0.5f, 0.5f, 0.1f, 1000.0f);
	projMatrix->perspective(45.0f, ratio, 0.1f, 1000.0f);
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

void GLWidget::mousePressEvent(QMouseEvent *e)
{
	float xFactor = width / 256.0f;	//256 = N_X
	float yFactor = height / 256.0f;	//256 = N_Y
	int x = e->x() / xFactor;
	int y = e->y() / yFactor;
	QMessageBox::information(this, "Coords", QString("X: %1 Y: %2").arg(x).arg(y), QDialogButtonBox::Ok);
}


