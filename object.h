#ifndef OBJECT_H
#define OBJECT_H

#include <QOpenGLFunctions_3_3_Core>
#include <QMatrix4x4>

#include "glwidget.h"
#include "cuglbuffer.h"
#include "texture.h"

class Object
{
public:
	Object(QString name, int instances, std::vector<CUGLBuffer*> buffers, Texture *texture, Shader *shader);
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

private:
	QOpenGLFunctions_3_3_Core* glFuncs;

	QMatrix4x4 modelMatrix;
	GLuint vao, mLoc, vLoc, pLoc;
	int fbo;
	bool instanced;
};

#endif // OBJECT_H


