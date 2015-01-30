#include "GLContext.h"
#include <map>

std::map<GLFWwindow*, GLContext*> windowList;

void glfwResizeCB(GLFWwindow *window, int width, int height)
{
	GLContext *wnd;

	auto search = windowList.find(window);
	if(search != windowList.end())
	{
		wnd = search->second;
		wnd->Width(width);
		wnd->Height(height);
	}
	else
	{
		Logger::Log("Unable to locate window to resize");
	}

	glViewport(0, 0, width, height);
	//resize matrices
}

GLContext::GLContext(std::string title, unsigned int w, unsigned int h, bool full)
{
	initialised = false;
	wndTitle = title;
	width = w;
	height = h;
	fullscreen = full;

	GLFWmonitor *monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode *vidMode = glfwGetVideoMode(monitor);

	if(fullscreen)
		wndHND = glfwCreateWindow(vidMode->width, vidMode->height, wndTitle.c_str(), glfwGetPrimaryMonitor(), NULL);
	else
		wndHND = glfwCreateWindow(width, height, wndTitle.c_str(), NULL, NULL);

	if(!wndHND)
	{
		Logger::Log("Error: Could not create a window with GLFW3\n");
	}
	else
	{
		glfwMakeContextCurrent(wndHND);
		windowList.insert(std::make_pair(wndHND, this));
		initialised = true;
	}

	glfwSetWindowSizeCallback(wndHND, glfwResizeCB);
}

GLContext::~GLContext()
{
	delete wndHND;
}

void GLContext::GetWindowSize(int &w, int &h)
{
	if(initialised)
		glfwGetWindowSize(wndHND, &w, &h);
}

void GLContext::SetWindowSize(int w, int h)
{
	if(initialised)
	{
		width = w;
		height = h;
		glfwSetWindowSize(wndHND, width, height);
	}
}