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

bool CUGLBuffer::InitVBO(int *bufferCapacity, float **bufferData, unsigned int *bufferUsage,
							int *bufferIndex, int *attribSize, GLenum *bufferType, bool *normalised)
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
		glVertexAttribPointer((GLuint)bufferIndex, attribSize[i], bufferType[i], normalised[i], 0, 0);
		glEnableVertexAttribArray((GLuint)bufferIndex[i]);
	}

	return true;
}

GLuint *CUGLBuffer::GetVBO(int index)
{
	return &VBO[index];
}

GLuint *CUGLBuffer::GetAllVBO()
{
	return VBO;
}


