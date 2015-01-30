#pragma once

#include <GL\glew.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Utilities.h"

class Shader
{
public:
	Shader(const char *vsFile, const char *fsFile);
	~Shader();

	void Bind();
	void Unbind();

	GLuint ShaderID() const { return shaderID; }

private:
	bool Initialise(const char *vsFile, const char *fsFile);
	bool ReadShaderFile(const char *file, unsigned int &id);

	GLuint shaderID, vertID, fragID;
	bool initialised;
};

