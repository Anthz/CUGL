#include "Surface2D.h"

Surface2D::Surface2D(Shape s, Shader *sh)
{
	shape = s;
	shader = sh;
	Initialise();
}

Surface2D::~Surface2D()
{
}

void Surface2D::Update()
{

}

void Surface2D::Render()
{
	shader->Bind();
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	shader->Unbind();
}

void Surface2D::Initialise()
{
	if(shape == SQUARE)
	{
		float points[] = {
			-0.5f, 0.5f, 0.0f,
			0.5f, 0.5f, 0.0f,
			-0.5f, -0.5f, 0.0f,
			0.5f, 0.5f, 0.0f,
			0.5f, -0.5f, 0.0f,
			-0.5f, -0.5f, 0.0f,
		};

		vbo = 0;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 18 * sizeof(float), &points, GL_STATIC_DRAW);
	}
	if(shape == TRIANGLE)
	{
		float points[] = {
			0.0f, 0.5f, 0.0f,
			0.5f, -0.5f, 0.0f,
			-0.5f, -0.5f, 0.0f
		};

		vbo = 0;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), &points, GL_STATIC_DRAW);
	}
	if(shape == CIRCLE)
	{
		float points[] = {
			0.000000, 0.000000, 0.000000,
			1.000000, 0.000000, 0.000000,
			0.980785, 0.195090, 0.000000,
			0.923880, 0.382683, 0.000000,
			0.831470, 0.555570, 0.000000,
			0.707107, 0.707107, 0.000000,
			0.555570, 0.831470, 0.000001,
			0.382683, 0.923880, 0.000001,
			0.195090, 0.980785, 0.000001,
			0.000000, 1.000000, 0.000001,
			-0.195090, 0.980785, 0.000001,
			-0.382683, 0.923880, 0.000001,
			-0.555570, 0.831470, 0.000001,
			-0.707107, 0.707107, 0.000000,
			-0.831470, 0.555570, 0.000000,
			-0.923880, 0.382683, 0.000000,
			-0.980785, 0.195090, 0.000000,
			-1.000000, -0.000000, -0.000000,
			-0.980785, -0.195091, -0.000000,
			-0.923879, -0.382684, -0.000000,
			-0.831469, -0.555571, -0.000000,
			-0.707106, -0.707107, -0.000000,
			-0.555570, -0.831470, -0.000001,
			-0.382683, -0.923880, -0.000001,
			-0.195089, -0.980785, -0.000001,
			0.000001, -1.000000, -0.000001,
			0.195091, -0.980785, -0.000001,
			0.382684, -0.923879, -0.000001,
			0.555571, -0.831469, -0.000001,
			0.707108, -0.707106, -0.000000,
			0.831470, -0.555569, -0.000000,
			0.923880, -0.382682, -0.000000,
			0.980786, -0.195089, -0.000000
		};

		vbo = 0;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 99 * sizeof(float), &points, GL_STATIC_DRAW);
	}

	vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}
