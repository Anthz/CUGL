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

	float points[] = {
		-0.5f, -0.5f, 0.0f,
		-0.5f, 0.5f, 0.0f,
		0.5f, 0.5f, 0.0f,
		0.5f, 0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, -0.5f, 0.0f
	};

	timer = new CUTimer();
	shader = new Shader("Colour.vert", "Colour.frag");
	shader->Bind();
	vbo = new CUGLBuffer();
	vbo->InitVBO(6 * 3, points, GL_DYNAMIC_DRAW, 0, 3, GL_FLOAT, false);
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

	if(GLFW_PRESS == glfwGetKey(mainWND->Handle(), GLFW_KEY_F5))
	{
		if(ERRORCHECK(cudaGraphicsMapResources(1, vbo->CudaVBO(), 0)))
		{
			size_t size = vbo->BufferSize() * sizeof(float);

			ERRORCHECK(cudaGraphicsResourceGetMappedPointer((void**)&vbo->DeviceBuffer(), &size, *vbo->CudaVBO()));

			timer->Begin();

			Kernel::ExecuteKernel(vbo->DeviceBuffer(), dim3(100, 100, 100));

			timer->End();
			timer->ElapsedTime();

			ERRORCHECK(cudaGraphicsUnmapResources(1, vbo->CudaVBO(), 0));
		}
	}

	glfwPollEvents();
}

void CUGL::Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	shader->Bind();
	glBindBuffer(GL_ARRAY_BUFFER, vbo->vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	shader->Unbind();

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