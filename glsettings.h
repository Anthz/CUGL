#ifndef GLSETTINGS_H
#define GLSETTINGS_H

#include <QWidget>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QTabWidget>

#include "glgeneraltab.h"
#include "objecttab.h"
#include "glbuffertab.h"
#include "texturetab.h"
#include "controltab.h"
#include "object.h"
#include "cuglbuffer.h"
#include "texture.h"

class GLSettings : public QWidget
{
	Q_OBJECT

public:
	explicit GLSettings(QWidget* parent = 0);
	~GLSettings();

	void AddTexture(Texture *texList);
	void AddBuffer(CUGLBuffer *bufList);
	void AddObject(Object *objList);
	void AddTextures(std::vector<Texture*> *texList);
	void AddBuffers(std::vector<CUGLBuffer*> *bufList);
	void AddObjects(std::vector<Object*> *objList);
	virtual QSize minimumSizeHint() const override;
	virtual QSize sizeHint() const override;

	static std::vector<Object*> ObjectList;
	static std::vector<CUGLBuffer*> BufferList;
	static std::vector<Texture*> TextureList;

private:
	QVBoxLayout *mainLayout, *settingsLayout;
	QGroupBox* settingsGroup;
	QTabWidget* tabs;
	GLGeneralTab *genTab;
	ObjectTab *objTab;
	GLBufferTab *bufTab;
	TextureTab *texTab;
	//move control to bottom bar

signals:

public slots:
};

#endif // GLSETTINGS_H


