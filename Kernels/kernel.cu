#ifdef Q_OS_MAC
#include <OpenGL/gl.h>
#else
#include "Windows.h"
#include <GL/gl.h>
#endif
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "math_constants.h"
#include "helper_math.h"
#include "utilities.h"

#define N 4096
#define N_ITERS 10
#define BLOCK_SIZE 256
#define N_BLOCKS (N + BLOCK_SIZE - 1) / BLOCK_SIZE

__global__ void setup_rand(curandState *state){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, idx, 0, &state[idx]);
}

__global__
void PBO(uchar4 *texBuf)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	//int y = threadIdx.y + blockIdx.y * blockDim.y;
	//int offset = x + y * blockDim.x * gridDim.x;

	if(x < N)
	{
		uchar4 col = texBuf[x];
		unsigned char val = x % 255;

		if((int)col.x - val < 0)
			col.x = 255;
		col.x -= val;
		if((int)col.y - val < 0)
			col.y = 255;
		col.y -= val;
		if((int)col.z - val < 0)
			col.z = 255;
		col.z -= val;

		texBuf[x] = col;
	}
}

//UI option for iterations
void CUExecuteKernel(std::vector<void*> *params)	//std::vector<void*> *params, size_t instances, float dt
{
	//CUTimer t;
	//t.Begin();
	PBO << <N_BLOCKS, BLOCK_SIZE >> >((uchar4*)(params->at(0)));

	ERRORCHECK(cudaGetLastError());
	//t.End();
}