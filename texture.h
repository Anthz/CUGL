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
	Texture(QString name, QString path, QImage image, int width, int height, std::pair<GLenum, QString> target, std::pair<GLint, QString> minMagFilter, std::pair<GLint, QString> wrapMode, bool fbo);
	Texture(QString name, int width, int height, std::pair<GLenum, QString> target, std::pair<GLint, QString> minMagFilter, std::pair<GLint, QString> wrapMode, bool fbo);
	~Texture();

	void Bind();
	void Unbind();

	QString Name() const { return name; }
	QString DataPath() const { return dataPath; }
	QImage Image() const { return image; }	//pass ref	
	QSize ImageSize() const { return imageSize; }
	std::pair<GLenum, QString> Target() const { return target; }
	std::pair<GLenum, QString> MinMagFilter() const { return minMagFilter; }
	std::pair<GLenum, QString> WrapMode() const { return wrapMode; }
	bool FBO() const { return fbo; }
	int FBOID() const { return fboID; }
	int PBO() const { return pbo; }
	const void *Data() const { return data; }


	void Name(QString val) { name = val; }
	void DataPath(QString val) { dataPath = val; }
	void Image(QImage val) { image = val; }
	void ImageSize(QSize val) { imageSize = val; }
	void Target(std::pair<GLenum, QString> val) { target = val; }
	void MinMagFilter(std::pair<GLenum, QString> val) { minMagFilter = val; }
	void WrapMode(std::pair<GLenum, QString> val) { wrapMode = val; }
	void FBO(int val) { fbo = val; }	
	void PBO(int val) { pbo = val; }
private:
	QString name, dataPath;
	GLuint texID;
	QImage image;
	QSize imageSize;
	const void *data;
	std::pair<GLenum, QString> target;
	std::pair<GLenum, QString> minMagFilter, wrapMode;
	bool fbo;
	int fboID;
	int pbo;

	QOpenGLFunctions_3_3_Core *glFuncs;
};

