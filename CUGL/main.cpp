#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <iostream>
#include <sstream>

#include "Utilities.h"
#include "CUGL.h"

void glfwErrorCB(int error, const char* desc)
{
	Logger::Log("GLFW Error " + std::to_string(error) + ": " + desc);
}

int main(int argc, char **argv)
{
	CUGL app;
	Logger::InitLogger();

	glfwSetErrorCallback(glfwErrorCB);

	if(!glfwInit())
	{
		Logger::Log("Error: Could not initialise GLFW3\n");
		return 1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_SAMPLES, 4);

	app.CreateWND("CUGL", 800, 600, false);

	glewExperimental = GL_TRUE;
	glewInit();

	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* version = glGetString(GL_VERSION);

	std::stringstream ss;

	ss << "Renderer: " << renderer;

	Logger::Log(ss.str());

	ss.str(std::string());
	ss << "OpenGL Version: " << version;

	Logger::Log(ss.str());

	app.Initialise();
	app.Run();

	glfwTerminate();
	return 0;
}
