#ifndef OBJECT_H
#define OBJECT_H

#include <QOpenGLFunctions_3_3_Core>
#include <QMatrix4x4>

#include "glwidget.h"
#include "cuglbuffer.h"
#include "texture.h"
#include "savable.h"

class Object : public Savable
{
public:
	Object(QString name, int instances, std::vector<int> *bufferIDs, int textureID, int shaderID);
	~Object();

	void Draw(GLenum drawMode, bool wireframe);
	void Move(QVector3D v);

	QString name;
	int instances;
	std::vector<CUGLBuffer*> buffers;
	Texture *texture;
	Shader *shader;

	int FBO() const { return fbo; }
	void FBO(int val) { fbo = val; }

	virtual void Save(QTextStream *output, std::vector<QString> *varList) override;

private:
	void GetBuffers(std::vector<int> *bufferIDs);
	void GetTexture(int texID);
	void GetShader(int shaderID);

	QOpenGLFunctions_3_3_Core* glFuncs;

	QMatrix4x4 modelMatrix;
	GLuint vao, mLoc, vLoc, pLoc;
	std::vector<int> bufferIDs;	//IDs = location within static vectors (not GL IDs)
	int textureID, shaderID, indicesID, fbo;
	bool instanced, indexed;
};

#endif // OBJECT_H