#ifndef CUGLBUFFER_H
#define CUGLBUFFER_H

#ifdef Q_OS_MAC
#include <OpenGL/gl.h>
#else
#include "Windows.h"
#include <GL/gl.h>
#endif
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_gl_interop.h>
#include <QOpenGLFunctions_3_3_Core>
#include <QImage>

#include "glwidget.h"

class CUGLBuffer
{
public:
	CUGLBuffer(QString name, int capacity, std::pair<GLenum, QString> target, QString data, std::pair<GLenum, QString> usage, std::pair<QString, int> attribID, int attribCapacity, std::tuple<GLenum, QString, int> type, bool norm);
	CUGLBuffer(QString name, int capacity, std::pair<GLenum, QString> target, void *data, std::pair<GLenum, QString> usage, std::pair<QString, int> attribID, int attribCapacity, std::tuple<GLenum, QString, int> type, bool norm);
	~CUGLBuffer();

	static void* RegisterBuffer(GLuint buf);
	static void UnregisterBuffer(void* res);
	static void* MapResource(void* res);
	static void UnmapResource(void* res);

	void Bind();
	bool InitVBO();

	void *CudaBuf() const { return cudaBuf; }

	QString bName, bDataPath;
	void *bData;
	int bCap, bSize, aSize;

	std::pair<GLenum, QString> bTarget, bUsage;
	std::pair<QString, int> aID;
	std::tuple<GLenum, QString, int> bType;
	bool norm;

private:
	void InitTex();
	bool LoadData();
	
	QOpenGLFunctions_3_3_Core* glFuncs;
	GLuint buf, tex;
	void *cudaBuf;
	QImage img;
	QSize texSize;

};

#endif // CUGLBUFFER_H


