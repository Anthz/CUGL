#include "DrawableSurface.h"

DrawableSurface::DrawableSurface(Shape s, Shader *sh, dim3 dimentions)
{
	shape = s;
	shader = sh;
	dims = dimentions;
	vertCount = 0;
	Initialise();
}

DrawableSurface::~DrawableSurface()
{
}

void DrawableSurface::Update()
{

}

void DrawableSurface::Render()
{
	shader->Bind();
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, vertCount);
	shader->Unbind();
}

void DrawableSurface::Initialise()
{
	vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	if(shape == SQUARE)
	{
		float points[] = {
			-0.5f, -0.5f, 0.0f,
			-0.5f, 0.5f, 0.0f,
			0.5f, 0.5f, 0.0f,
			0.5f, 0.5f, 0.0f,
			0.5f, -0.5f, 0.0f,
			-0.5f, -0.5f, 0.0f
		};
		
		vertCount = 6;
		vbo = 0;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, (vertCount * 3) * sizeof(float), &points, GL_STATIC_DRAW);
	}
	if(shape == TRIANGLE)
	{
		float points[] = {
			0.0f, 0.5f, 0.0f,
			0.5f, -0.5f, 0.0f,
			-0.5f, -0.5f, 0.0f
		};

		vertCount = 3;
		vbo = 0;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, (vertCount * 3) * sizeof(float), &points, GL_STATIC_DRAW);
	}
	if(shape == CIRCLE)
	{
		vertCount = 33;
		float points[] = {
			0.000000, 0.000000, 0.000000,
			1.000000, 0.000000, 0.000000,
			0.980785, 0.195090, 0.000000,
			0.923880, 0.382683, 0.000000,
			0.831470, 0.555570, 0.000000,
			0.707107, 0.707107, 0.000000,
			0.555570, 0.831470, 0.000000,
			0.382683, 0.923880, 0.000000,
			0.195090, 0.980785, 0.000000,
			0.000000, 1.000000, 0.000000,
			-0.195090, 0.980785, 0.000000,
			-0.382683, 0.923880, 0.000000,
			-0.555570, 0.831470, 0.000000,
			-0.707107, 0.707107, 0.000000,
			-0.831470, 0.555570, 0.000000,
			-0.923880, 0.382683, 0.000000,
			-0.980785, 0.195090, 0.000000,
			-1.000000, -0.000000, -0.000000,
			-0.980785, -0.195091, -0.000000,
			-0.923879, -0.382684, -0.000000,
			-0.831469, -0.555571, -0.000000,
			-0.707106, -0.707107, -0.000000,
			-0.555570, -0.831470, -0.000000,
			-0.382683, -0.923880, -0.000000,
			-0.195089, -0.980785, -0.000000,
			0.000001, -1.000000, -0.000000,
			0.195091, -0.980785, -0.000000,
			0.382684, -0.923879, -0.000000,
			0.555571, -0.831469, -0.000000,
			0.707108, -0.707106, -0.000000,
			0.831470, -0.555569, -0.000000,
			0.923880, -0.382682, -0.000000,
			0.980786, -0.195089, -0.000000
		};

		vbo = 0;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, (vertCount * 3) * sizeof(float), &points, GL_STATIC_DRAW);
	}

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
	glDeleteBuffers(1, &vbo);
<<<<<<< HEAD
=======

	GenTexture();
>>>>>>> origin/master
}

void DrawableSurface::GenTexture()
{
	cudaError_t e;

	glGenTextures(1, &texID);
	if(dims.x > 1)
	{
		if(dims.y > 1)
		{
			if(dims.z > 1)
			{
				glBindTexture(GL_TEXTURE_3D, texID);
				glBindTexture(GL_TEXTURE_3D, texID);
				{
					glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
					glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
					glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
					glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
					glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

					glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, dims.x, dims.y, dims.z, 0, GL_RGBA, GL_FLOAT, NULL);
				}
				glBindTexture(GL_TEXTURE_3D, 0);
				e = cudaGraphicsGLRegisterImage(&cudaImageResource, texID, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
			}
			else
			{
				glBindTexture(GL_TEXTURE_2D, texID);
				glBindTexture(GL_TEXTURE_2D, texID);
				{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, dims.x, dims.y, 0, GL_RGBA, GL_FLOAT, NULL);
				}
				glBindTexture(GL_TEXTURE_2D, 0);
				e = cudaGraphicsGLRegisterImage(&cudaImageResource, texID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
			}
		}
		else
		{
			glBindTexture(GL_TEXTURE_1D, texID);
			glBindTexture(GL_TEXTURE_1D, texID);
			{
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

				glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, dims.x, 0, GL_RGBA, GL_FLOAT, NULL);
			}
			glBindTexture(GL_TEXTURE_1D, 0);
			e = cudaGraphicsGLRegisterImage(&cudaImageResource, texID, GL_TEXTURE_1D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		}
	}

	ERRORCHECK(e);
}

void DrawableSurface::MapTexture()
{
	ERRORCHECK(cudaGraphicsMapResources(1, &cudaImageResource, 0));
	ERRORCHECK(cudaGraphicsSubResourceGetMappedArray(&texArray, cudaImageResource, 0, 0));

	cudaResourceDesc resourceDesc;
	memset(&resourceDesc, 0, sizeof(cudaResourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
	resourceDesc.res.array.array = texArray;

	tex = 0;
	cudaCreateSurfaceObject(&tex, &resourceDesc);
}

