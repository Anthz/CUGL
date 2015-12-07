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
	tabs->addTab(new GLGeneralTab(this), "General");
	tabs->addTab(new ObjectTab(this), "Objects");
	tabs->addTab(new GLBufferTab(this), "Buffers");
	tabs->addTab(new TextureTab(this), "Textures");
	tabs->addTab(new ControlTab(this), "Controls");
	settingsLayout->addWidget(tabs);
	settingsGroup->setLayout(settingsLayout);
	mainLayout->addWidget(settingsGroup);

	setLayout(mainLayout);
}

GLSettings::~GLSettings()
{
	delete tabs;
	delete settingsLayout;
	delete settingsGroup;

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

QSize GLSettings::minimumSizeHint() const
{
	return QSize(500, 350);
}

QSize GLSettings::sizeHint() const
{
	return QSize((int)static_cast<QWidget*>(parent())->width() / 3, 350);
}