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
#include "savable.h"

class CUGLBuffer : public Savable
{
public:
	CUGLBuffer(QString name, int capacity, QString target, QString data, QString dataPath, QString usage, QString attribID, int attribCapacity, QString type, bool norm, bool perInst);
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
	bool PerInstance() const { return perInstance; }

	virtual void Save(QTextStream *output, std::vector<QString> *varList) override;

	GLuint bufID;
	QString bName, bDataType, bDataPath;
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
	bool LoadData(QString dataType);
	void GetGLTarget(QString targetString);
	void GetGLUsage(QString usageString);
	void GetAttribID(QString attribString);
	void GetGLType(QString typeString);

	QOpenGLFunctions_3_3_Core* glFuncs;
	GLuint tex;
	void *cudaBuf;
	QImage img;
	QSize texSize;
	bool cuda, perInstance;
	int paramID;

	template <typename T>
	inline void RandomData(T *data, int const& n, float const& min, float const& max)
	{
		for(int i = 0; i < n; i++)
		{
			data[i] = (max - min) * (rand() / (float)RAND_MAX) + min;
		}
	}

	//parse file of type T
	//storing 
	template <typename T>
	inline void ParseFile(T *data, const char& delim = ' ')
	{
		std::ifstream in(bDataPath.toStdString());
		std::string s = "|";
		std::vector<QString> elems;
		int counter = 0;

		getline(in, s);
		while(s.size() != 0)
		{
			split(s, delim, elems);
			for(int i = 0; i < s.size(); ++i)
			{
				data[counter + i] = elems.at(i).toDouble();
			}
			counter += s.size();
			elems.clear();
			getline(in, s);
		}
	}
};

#endif // CUGLBUFFER_H


