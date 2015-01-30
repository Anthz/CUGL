#pragma once

#include <gl/glew.h>
#include <GLFW\glfw3.h>
#include "Shader.h"

class Surface2D
{
public:
	enum Shape
	{
		SQUARE,
		TRIANGLE,
		CIRCLE
	};


	Surface2D(Shape s, Shader *sh);
	~Surface2D();

	void Update();
	void Render();

private:
	void Initialise();

	Shape shape;
	Shader *shader;
	GLuint vbo, vao;
};

