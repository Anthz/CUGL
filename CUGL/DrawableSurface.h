#pragma once

#include <gl/glew.h>
#include <GLFW\glfw3.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "Shader.h"

class DrawableSurface
{
public:
	enum Shape
	{
		SQUARE,
		TRIANGLE,
		CIRCLE
	};


	DrawableSurface(Shape s, Shader *sh, dim3 dimentions);
	~DrawableSurface();

	void Update();
	void Render();
	void MapTexture();

	cudaSurfaceObject_t * Tex() const { return tex; }
private:
	void Initialise();
	void GenTexture();

	cudaGraphicsResource *cudaImageResource;
	cudaArray_t texArray;
	cudaSurfaceObject_t *tex;

	Shape shape;
	Shader *shader;
	GLuint vbo, vao, texID;
	dim3 dims;
	unsigned int vertCount;
};

