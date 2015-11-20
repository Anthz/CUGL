#pragma once

#include <QImage>
#include <QImageReader>
#include <QMessageBox>
#include <QOpenGLFunctions_3_3_Core>
#include <QtGui/qopengl.h>

#include "glwidget.h"

class Texture
{
public:
	Texture(QString name, QString path, GLenum target = GL_TEXTURE_2D, GLint minMagFilter = GL_LINEAR, GLint wrapMode = GL_CLAMP_TO_EDGE);
	~Texture();

	void Bind();
	void Unbind();

	QString Name() const { return name; }
	void Name(QString val) { name = val; }

	QImage Image() const { return image; }	//pass ref	
	QSize ImageSize() const { return imageSize; }
	GLenum Target() const { return target; }
	GLint MinMagFilter() const { return minMagFilter; }
	GLint WrapMode() const { return wrapMode; }

private:
	QString name;
	GLuint texID;
	QImage image;
	QSize imageSize;
	const void *data;
	GLenum target;
	GLint minMagFilter, wrapMode;

	QOpenGLFunctions_3_3_Core *glFuncs;
};

