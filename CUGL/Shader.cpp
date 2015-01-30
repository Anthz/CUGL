#include "Shader.h"

Shader::Shader(const char *vsFile, const char *fsFile)
{
	initialised = Initialise(vsFile, fsFile);
}

Shader::~Shader()
{
	glDetachShader(shaderID, vertID);
	glDetachShader(shaderID, fragID);
	glDeleteShader(vertID);
	glDeleteShader(fragID);
	glDeleteShader(shaderID);
}

bool Shader::Initialise(const char *vsFile, const char *fsFile)
{
	vertID = glCreateShader(GL_VERTEX_SHADER);
	fragID = glCreateShader(GL_FRAGMENT_SHADER);

	ReadShaderFile(vsFile, vertID);
	ReadShaderFile(fsFile, fragID);

	glCompileShader(vertID);
	glCompileShader(fragID);

	shaderID = glCreateProgram();
	glAttachShader(shaderID, vertID);
	glAttachShader(shaderID, fragID);
	glLinkProgram(shaderID);

	return true;
}

void Shader::Bind()
{
	glUseProgram(shaderID);
}

void Shader::Unbind()
{
	glUseProgram(0);
}

bool Shader::ReadShaderFile(const char *file, unsigned int &id)
{
	std::ifstream f;
	f.open(file);

	if(!f.good())
	{
		Logger::Log("Failed to open file: " + *file);
		return false;
	}

	std::stringstream ss;
	ss << f.rdbuf();
	f.close();

	std::string s = ss.str();
	const char* c = s.c_str();

	glShaderSource(id, 1, &c, 0);
}
