#include "shader.h"

static const char* vertexShaderSource =
"attribute highp vec3 aPos;\n"
"attribute highp vec3 iPos;\n"
"attribute mediump vec2 aUV;\n"
"uniform highp mat4 uModelMatrix;\n"
"uniform highp mat4 uViewMatrix;\n"
"uniform highp mat4 uProjMatrix;\n"
"varying mediump vec2 texCoords;\n"
"void main() {\n"
"   texCoords = aUV;"
"   gl_Position = uProjMatrix * uViewMatrix * uModelMatrix * vec4(aPos + iPos, 1.0);\n"
"}\n";

static const char* fragmentShaderSource =
"uniform sampler2D texture;\n"
"varying mediump vec2 texCoords;\n"
"void main() {\n"
"   gl_FragColor = texture2D(texture, texCoords);\n"
"}\n";

Shader::Shader(QString name, QString vertPath, QString fragPath)
{
	this->name = name;

	bool result;
	program = new QOpenGLShaderProgram();
	result = program->addShaderFromSourceFile(QOpenGLShader::Vertex, vertPath);
	result = program->addShaderFromSourceFile(QOpenGLShader::Fragment, fragPath);
	//result = program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
	//result = program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
	result = program->link();

	if(!result)
	{
		qWarning() << "Shader could not be initialised";
		exit(1);
	}
}

Shader::~Shader()
{
	delete program;
}

void Shader::Bind()
{
	program->bind();
}

void Shader::Release()
{
	program->release();
}

GLuint Shader::GetAttribLoc(const QString name)
{
	int id = program->attributeLocation(name);
	attributes.push_back(std::pair<QString, int>(name, id));
	return id;
}

GLuint Shader::GetUniformLoc(const QString name)
{
	return program->uniformLocation(name);
}

void Shader::SetUniform(int id, const QMatrix4x4 matrix)
{
	program->setUniformValue(id, matrix);
}

void Shader::SetUniform(const char *name, const QMatrix4x4 matrix)
{
	program->setUniformValue(name, matrix);
}

