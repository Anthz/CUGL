#pragma once

#include "Utilities.h"
#include "GLContext.h"
#include "Shader.h"

class CUGL
{
public:
	CUGL();
	~CUGL();

	bool Initialise();
	void Run();

	void Update();
	void Render();

	void CreateWND(std::string title, unsigned int w, unsigned int h, bool full);
	void Resize(GLFWwindow *window, int width, int height);

private:
	void UpdateFPS();

	GLContext *mainWND;
	Shader *shader;

	GLuint vbo, vao;
};
