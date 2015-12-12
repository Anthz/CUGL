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
#include <fstream>
#include <iostream>
#include <sstream>

#include "glwidget.h"
#include "utilities.h"

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
	void Unbind();
	bool InitVBO();

	void *CudaBuf() const { return cudaBuf; }
	bool Cuda() const { return cuda; }
	void Cuda(bool val) { cuda = val; }
	int ParamID() const { return paramID; }
	void ParamID(int val) { paramID = val; }

	GLuint bufID;
	QString bName, bDataPath;
	void *bData;
	int bCap, bSize, aSize;

	std::pair<GLenum, QString> bTarget, bUsage;
	std::pair<QString, int> aID;
	std::tuple<GLenum, QString, int> bType;
	bool norm;
private:
	void InitTex();
	void Randomise(float *data, int n);
	void ParseFile(float *data);
	bool LoadData();

	QOpenGLFunctions_3_3_Core* glFuncs;
	GLuint tex;
	void *cudaBuf;
	QImage img;
	QSize texSize;
	bool cuda;
	int paramID;

};

#endif // CUGLBUFFER_H


