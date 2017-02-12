#pragma once

#include <QImage>
#include <QImageReader>
#include <QMessageBox>
#include <QOpenGLFunctions_3_3_Core>
#include <QtGui/qopengl.h>

#include "glwidget.h"
#include "savable.h"

class Texture : public Savable
{
public:
	Texture(QString name, QString path, int width, int height, QImage::Format fmt, QString targetString, QString filterString, QString wrapString, bool fbo, int pbo = -1);
	~Texture();

	void Bind();
	void Unbind();

	QString Name() const { return name; }
	QString DataPath() const { return dataPath; }
	QImage Image() const { return image; }	//pass ref	
	QSize ImageSize() const { return imageSize; }
	std::pair<GLenum, QString> Target() const { return target; }
	std::pair<GLint, QString> MinMagFilter() const { return minMagFilter; }
	std::pair<GLint, QString> WrapMode() const { return wrapMode; }
	std::pair<GLint, GLenum> GLFmt() const { return glFmt; }
	bool FBO() const { return fbo; }
	int FBOID() const { return fboID; }
	int PBO() const { return pbo; }
	const void *Data() const { return data; }
	unsigned int FormatCount();

	void Name(QString val) { name = val; }
	void DataPath(QString val) { dataPath = val; }
	void Image(QImage val) { image = val; }
	void ImageSize(QSize val) { imageSize = val; }
	void Target(std::pair<GLenum, QString> val) { target = val; }
	void MinMagFilter(std::pair<GLint, QString> val) { minMagFilter = val; }
	void WrapMode(std::pair<GLint, QString> val) { wrapMode = val; }
	void GLFmt(std::pair<GLint, GLenum> val) { glFmt = val; }
	void FBO(int val) { fbo = val; }
	void PBO(int val) { pbo = val; }

	virtual void Save(QTextStream *output, std::vector<QString> *varList) override;	//seperate save/load funcs or override base savable method? could use vector or delimed string, then same param
	//static void Load(QTextStream *input, std::vector<Texture*> *outList);

private:
	void GetGLFormat();
	void GetGLTarget(QString targetString);
	void GetGLMinMagFilter(QString filterString);
	void GetGLWrapMode(QString wrapString);

	QString name, dataPath;
	GLuint texID;
	QImage image;
	QSize imageSize;
	const void *data;
	std::pair<GLenum, QString> target;
	std::pair<GLint, QString> minMagFilter, wrapMode;
	std::pair<GLint, GLenum> glFmt;
	bool fbo;
	int fboID;
	int pbo;

	QOpenGLFunctions_3_3_Core *glFuncs;
};

