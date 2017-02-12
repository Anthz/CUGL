#include "glsettings.h"

std::vector<Object*> GLSettings::ObjectList;
std::vector<CUGLBuffer*> GLSettings::BufferList;
std::vector<Texture*> GLSettings::TextureList;

GLSettings::GLSettings(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QVBoxLayout();
	settingsLayout = new QVBoxLayout();

	settingsGroup = new QGroupBox("OpenGL Settings");
	tabs = new QTabWidget();
	tabs->addTab(genTab = new GLGeneralTab(this), "General");
	tabs->addTab(objTab = new ObjectTab(this), "Objects");
	tabs->addTab(bufTab = new GLBufferTab(this), "Buffers");
	tabs->addTab(texTab = new TextureTab(this), "Textures");
	tabs->addTab(new ControlTab(this), "Controls");
	settingsLayout->addWidget(tabs);
	settingsGroup->setLayout(settingsLayout);
	mainLayout->addWidget(settingsGroup);

	setLayout(mainLayout);
}

GLSettings::~GLSettings()
{
	for each (Object *o in ObjectList)
	{
		delete o;
	}

	for each (CUGLBuffer *b in BufferList)
	{
		delete b;
	}

	for each (Texture *t in TextureList)
	{
		delete t;
	}

	delete mainLayout;
}

void GLSettings::AddTexture(Texture *tex)
{
	TextureList.push_back(tex);
	texTab->AddToList(tex);
}

void GLSettings::AddBuffer(CUGLBuffer *buf)
{
	BufferList.push_back(buf);
	bufTab->AddToTable(buf);
}

void GLSettings::AddObject(Object *obj)
{
	ObjectList.push_back(obj);
	objTab->AddToTable(obj);
}

void GLSettings::AddTextures(std::vector<Texture*> *texList)
{
	for each (Texture* t in *texList)
	{
		TextureList.push_back(t);
		texTab->AddToList(t);
	}
}

void GLSettings::AddBuffers(std::vector<CUGLBuffer*> *bufList)
{
	for each (CUGLBuffer* b in *bufList)
	{
		BufferList.push_back(b);
		bufTab->AddToTable(b);
	}
}

void GLSettings::AddObjects(std::vector<Object*> *objList)
{
	for each (Object* o in *objList)
	{
		ObjectList.push_back(o);
		objTab->AddToTable(o);
	}
}

QSize GLSettings::minimumSizeHint() const
{
	return QSize(500, 350);
}

QSize GLSettings::sizeHint() const
{
	return QSize((int)static_cast<QWidget*>(parent())->width() / 3, 350);
}