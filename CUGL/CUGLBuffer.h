#pragma once
#include <GL\glew.h>
#include <cuda_gl_interop.h>
#include "Utilities.h"
class CUGLBuffer
{
public:
	CUGLBuffer(int numBuffers);
	~CUGLBuffer();

	void GetVBO(int index);

	unsigned int Count() const { return count; }

private:
	bool InitVBO(int *bufferCapacity, float *bufferData, int *bufferUsage);

	GLuint *VBO;
	int count, *bufferSize;

};

