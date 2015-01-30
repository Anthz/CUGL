#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <functional>

#include "Utilities.h"

class GLContext
{
public:
	GLContext(std::string title, unsigned int w, unsigned int h, bool full);
	~GLContext();

	void GetWindowSize(int &w, int &h);
	void SetWindowSize(int w, int h);

	GLFWwindow *Handle() const { return wndHND; }

	int Height() const { return height; }
	void Height(int val) { height = val; }
	int Width() const { return width; }
	void Width(int val) { width = val; }

	std::string Title() const { return wndTitle; }

private:
	GLFWwindow *wndHND;
	std::string wndTitle;
	int width, height;
	bool initialised, fullscreen;
};

