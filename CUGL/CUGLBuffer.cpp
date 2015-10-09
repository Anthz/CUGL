#include "CUGLBuffer.h"

CUGLBuffer::CUGLBuffer(int num)
{
	VBO = new GLuint[num];
	glGenBuffers(num, VBO);

	vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	bSize = new int[num];
	aIndex = new int[num];
	aSize = new int[num];
	bType = new GLenum[num];
	bUsage = new GLenum[num];
	norm = new bool[num];
}

CUGLBuffer::~CUGLBuffer()
{
	delete[] VBO;
	delete cudaVBO;

	delete[] bSize;
	delete[] aIndex;
	delete[] aSize;
	delete[] bType;
	delete[] bUsage;
	delete[] norm;
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
bool CUGLBuffer::InitVBO(int bufferID, int bufferCapacity, float *bufferData, GLenum bufferUsage,
	int attribIndex, int attribSize, GLenum bufferType, bool normalised)
{
	bSize[bufferID] = bufferCapacity;
	aIndex[bufferID] = attribIndex;
	aSize[bufferID] = attribSize;
	bType[bufferID] = bufferType;
	bUsage[bufferID] = bufferUsage;
	norm[bufferID] = normalised;

	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, bSize[bufferID] * sizeof(float), bufferData, bUsage[bufferID]);
	glVertexAttribPointer((GLuint)aIndex[bufferID], aSize[bufferID], bType[bufferID], norm[bufferID], 0, 0);
	glEnableVertexAttribArray((GLuint)aIndex[bufferID]);

	cudaVBO = (cudaGraphicsResource_t*)malloc(sizeof(cudaGraphicsResource_t));

	RegisterBuffer();

	return true;
}

bool CUGLBuffer::RegisterBuffer()
{
	if(!ERRORCHECK(cudaGraphicsGLRegisterBuffer(cudaVBO, *VBO, cudaGraphicsRegisterFlagsNone)))
	{
		return false;
	}

	//ERRORCHECK(cudaMalloc((void**)d_Buffer, sizeof(float) * bSize));
	//cudaMemset(d_Buffer, 0, sizeof(T) * bSize);

	Logger::Log("Registered GL buffer with CUDA.");
	return true;
}