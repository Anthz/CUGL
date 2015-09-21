#include "CUGLBuffer.h"

CUGLBuffer::CUGLBuffer()
{
	VBO = new GLuint;
	glGenBuffers(1, VBO);
}

CUGLBuffer::~CUGLBuffer()
{
	delete VBO;
}

/************************************************************************
* index: buffer id
* bufferCapacity: total number of elements per buffer
* bufferData: data to be copied to buffer
* bufferUsage: usage (GL_STREAM_DRAW, GL_STATIC_DRAW, GL_DYNAMIC_DRAW...)
* attribIndex: id of attrib in shader
* attribSize: number of elements per attrib
* bufferType: type of buffer elements (GL_FLOAT...)
* normalised: T/F
************************************************************************/
bool CUGLBuffer::InitVBO(int bufferCapacity, float *bufferData, GLenum bufferUsage,
	int attribIndex, int attribSize, GLenum bufferType, bool normalised)
{
	bSize = bufferCapacity;
	aIndex = attribIndex;
	aSize = attribSize;
	bType = bufferType;
	bUsage = bufferUsage;
	norm = normalised;

	vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, bSize * sizeof(float), bufferData, bUsage);
	glVertexAttribPointer((GLuint)aIndex, aSize, bType, norm, 0, 0);
	glEnableVertexAttribArray((GLuint)aIndex);

	RegisterBuffer();

	return true;
}

bool CUGLBuffer::RegisterBuffer()
{
	if(cudaGraphicsGLRegisterBuffer(cudaVBO, *VBO, cudaGraphicsMapFlagsNone) != cudaSuccess)
	{
		Logger::Log("Failed to register GL buffer with CUDA.");
		return false;
	}

	cudaMalloc((void**)&d_Buffer, sizeof(float) * bSize);
	//cudaMemset(d_Buffer, 0, sizeof(T) * bSize);

	Logger::Log("Registered GL buffer with CUDA.");
	return true;
}