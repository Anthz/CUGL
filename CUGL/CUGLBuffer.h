#pragma once
#include <GL\glew.h>
#include <cuda_gl_interop.h>
#include "Utilities.h"

class CUGLBuffer
{
public:
	CUGLBuffer();
	~CUGLBuffer();

	bool InitVBO(int bufferCapacity, float *bufferData, GLenum bufferUsage,
		int bufferIndex, int attribSize, GLenum bufferType, bool normalised);
	bool RegisterBuffer();

	(float*)& DeviceBuffer() { return d_Buffer; }
	cudaGraphicsResource_t *CudaVBO() const { return cudaVBO; }
	GLuint *GetVBO() const { return VBO; }

	int BufferSize() const { return bSize; }
	int AttribSize() const { return aSize; }
	int AttribIndex() const { return aIndex; }
	GLenum BufferUsage() const { return bUsage; }
	GLenum BufferType() const { return bType; }
	bool Normalised() const { return norm; }

	GLuint vao;

private:
	float *d_Buffer;
	cudaGraphicsResource_t *cudaVBO;
	GLuint *VBO;
	int bSize, aIndex, aSize;
	GLenum bType, bUsage;
	bool norm;
};