#include "CUGL.h"

CUGL::CUGL()
{
}

CUGL::~CUGL()
{
}

bool CUGL::Initialise()
{
	glClearColor(0.4f, 0.6f, 0.9f, 0.0f);

	shader = new Shader("Colour.vert", "Colour.frag");
	drawableSurface = new Surface2D(Surface2D::CIRCLE, shader);

	return true;
}

void CUGL::Run()
{
	glEnable(GL_DEPTH);
	glDepthFunc(GL_LESS);

	while(!glfwWindowShouldClose(mainWND->Handle()))
	{
		Update();
		Render();
	}
}

void CUGL::Update()
{
	UpdateFPS();

	if(GLFW_PRESS == glfwGetKey(mainWND->Handle(), GLFW_KEY_ESCAPE)) 
	{
		glfwSetWindowShouldClose(mainWND->Handle(), 1);
	}

	glfwPollEvents();
}

void CUGL::Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawableSurface->Render();

	glfwSwapBuffers(mainWND->Handle());
}

void CUGL::CreateWND(std::string title, unsigned int w, unsigned int h, bool full)
{
	mainWND = new GLContext(title, w, h, full);
}

void CUGL::UpdateFPS()
{
	static double prevTime = glfwGetTime();
	static int frameCount;
	double elapTime = glfwGetTime() - prevTime;

	if(elapTime >= 0.25)	//update every 1/4 second
	{
		prevTime = glfwGetTime();
		int fps = ((double)frameCount / elapTime) + 0.5;
		std::string s = mainWND->Title() + " | FPS: " + std::to_string(fps);
		glfwSetWindowTitle(mainWND->Handle(), s.c_str());
		frameCount = 0;
	}

	frameCount++;
}
