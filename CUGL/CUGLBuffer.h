#pragma once
#include <GL\glew.h>
#include <cuda_gl_interop.h>
#include "Utilities.h"
#include <vector>

class CUGLBuffer
{
public:
	CUGLBuffer(int num);
	~CUGLBuffer();

	bool InitVBO(int bufferID, int bufferCapacity, float *bufferData, GLenum bufferUsage,
		int bufferIndex, int attribSize, GLenum bufferType, bool normalised);
	bool RegisterBuffer();

	(float*)& DeviceBuffer() { return d_Buffer; }
	cudaGraphicsResource_t *CudaVBO() const { return cudaVBO; }

	int BufferSize(int index) const { return bSize[index]; }
	int AttribSize(int index) const { return aSize[index]; }
	int AttribIndex(int index) const { return aIndex[index]; }
	GLenum BufferUsage(int index) const { return bUsage[index]; }
	GLenum BufferType(int index) const { return bType[index]; }
	bool Normalised(int index) const { return norm[index]; }

	GLuint vao;

private:
	float *d_Buffer;
	cudaGraphicsResource_t *cudaVBO;
	GLuint *VBO;
	int *bSize, *aIndex, *aSize;
	GLenum *bType, *bUsage;
	bool *norm;
};