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

	return true;
}

void CUGL::Run()
{
	glEnable(GL_DEPTH);
	glDepthFunc(GL_LESS);

	float points[] = {
		0.0f, 0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, -0.5f, 0.0f
	};

	vbo = 0;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), &points, GL_STATIC_DRAW);

	vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

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
	shader->Bind();
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);
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
