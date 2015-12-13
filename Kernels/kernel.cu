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

inline
__device__ unsigned char countAliveCells(int *data, size_t x0, size_t x1, size_t x2,
	size_t y0, size_t y1, size_t y2) {
	return data[x0 + y0] + data[x1 + y0] + data[x2 + y0]
		+ data[x0 + y1] + data[x2 + y1]
		+ data[x0 + y2] + data[x1 + y2] + data[x2 + y2];
}

__global__
void GOL(uchar4 *texBuf, int *lifeBuf, int *output, size_t width, size_t height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if(offset < N)
	{
		size_t x1 = offset % width;
		size_t y1 = offset - x1;
		size_t y0 = (y1 + (width * height) - width) % (width * height);
		size_t y2 = (y1 + width) % (width * height);
		size_t x0 = (x1 + width - 1) % width;
		size_t x2 = (x1 + 1) % width;

		unsigned char aliveCells = countAliveCells(lifeBuf, x0, x1, x2, y0, y1, y2);
		output[y1 + x1] =
			aliveCells == 3 || (aliveCells == 2 && lifeBuf[x1 + y1]) ? 1 : 0;

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
		col.w = 255;

		texBuf[x] = col;
	}
}

//UI option for iterations
void CUExecuteKernel(std::vector<void*> *params)	//std::vector<void*> *params, size_t instances, float dt
{
	//CUTimer t;
	//t.Begin();
	GOL<<<N_BLOCKS, BLOCK_SIZE >> >((uchar4*)(params->at(0)));

	ERRORCHECK(cudaGetLastError());
	//t.End();
}