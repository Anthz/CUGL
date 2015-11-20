#ifndef SHADER_H
#define SHADER_H

#include <QOpenGLShaderProgram>

class Shader
{
public:
	Shader(QString name);
	~Shader();

	void Bind();
	void Release();
	GLuint GetAttribLoc(const QString name);
	GLuint GetUniformLoc(const QString name);
	void SetUniform(const char *name, const QMatrix4x4 matrix);

	QString name;

private:
	QOpenGLShaderProgram* program;
	std::vector<std::pair<QString, int>> attributes;

signals:

public slots:
};

#endif // SHADER_H


