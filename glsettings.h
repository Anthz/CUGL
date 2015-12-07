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

	virtual QSize minimumSizeHint() const override;
	virtual QSize sizeHint() const override;

	static std::vector<Object*> ObjectList;
	static std::vector<CUGLBuffer*> BufferList;
	static std::vector<Texture*> TextureList;

private:
	QVBoxLayout *mainLayout, *settingsLayout;
	QGroupBox* settingsGroup;
	QTabWidget* tabs;

signals:

public slots:
};

#endif // GLSETTINGS_H


