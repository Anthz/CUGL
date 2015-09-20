#pragma once
#include <GL\glew.h>
#include <cuda_gl_interop.h>
#include "Utilities.h"
class CUGLBuffer
{
public:
	CUGLBuffer(int numBuffers);
	~CUGLBuffer();

	GLuint *GetVBO(int index);
	GLuint *GetAllVBO();

	unsigned int Count() const { return count; }

private:
	bool InitVBO(int *bufferCapacity, float **bufferData, unsigned int *bufferUsage,
					int *bufferIndex, int *attribSize, GLenum *bufferType, bool *normalised);

	GLuint *VBO;
	int count, *bufferSize;

};

