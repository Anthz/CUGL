#include "CUGLBuffer.h"


CUGLBuffer::CUGLBuffer(int numBuffers)
{
	count = numBuffers;
	bufferSize = new int[count];
}

CUGLBuffer::~CUGLBuffer()
{
	delete[] VBO;
	delete[] bufferSize;
}

bool CUGLBuffer::InitVBO(int *bufferCapacity, float **bufferData, int *bufferUsage)
{
	if(sizeof(*bufferCapacity) != sizeof(int) * count)
	{
		if(sizeof(*bufferCapacity) == sizeof(int))
		{
			for(int i = 0; i < count; ++i)
			{
				bufferSize[i] = *bufferCapacity;
			}
		}
		else
		{
			Logger::Log("Capacity/Buffer mismatch. Either provide individual sizes or a single value.");
			return false;
		}
	}
	else
	{
		for(int i = 0; i < count; ++i)
		{
			bufferSize[i] = bufferCapacity[i];
		}
	}

	VBO = new GLuint[count];
	glGenBuffers(count, VBO);
	for(int i = 0; i < count; ++i)
	{
		glBindBuffer(GL_ARRAY_BUFFER, VBO[i]);
		glBufferData(GL_ARRAY_BUFFER, bufferCapacity[i], (GLvoid*)bufferData[i], bufferUsage[i]);
		glVertexAttribPointer((GLuint)0,)
	}

	return true;
}


