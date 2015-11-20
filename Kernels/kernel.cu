#include <stdio.h>
#ifdef Q_OS_MAC
#include <OpenGL/gl.h>
#else
#include "Windows.h"
#include <GL/gl.h>
#endif
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_gl_interop.h>
#include "math_constants.h"

//pass in dt
__global__ void particleKernel(float *pos, size_t instances)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = 3;
	int speed = 2.5f;
	float angle = (45.0f / instances) * i;
	float rads = angle * CUDART_PI / 180.0f;

	if(i < instances)
	{
			//x
			pos[i * stride + 0] += (speed * cosf(angle)); /*(i % 2 == 0) ?
			pos[i * stride + 0] + (0.5f * (i / 100.0f)):
			pos[i * stride + 0] - (0.5f * (i / 100.0f));*/

		//y
		pos[i * stride + 1] += (-speed * sinf(angle));//0.5f * (i / 100.0f);

		//z
		/*pos[i * stride + 2] = (i % 2 == 0) ?
			pos[i * stride + 2] + 1.0f * (i / 100.0f) :
			pos[i * stride + 2] - 1.0f * (i / 100.0f);*/

		if(pos[i * stride + 1] > 100.0f)
		{
			pos[i * stride + 0] = 0.0f;
			pos[i * stride + 1] = 0.0f;
			pos[i * stride + 2] = 0.0f;
		}
	}
}

void particleMovement(float* pos, size_t instances)
{
    int threadsPerBlock = 256;	//okay as long as instances < max threads/block
    int blocksPerGrid = (instances + threadsPerBlock - 1) / threadsPerBlock;

    particleKernel<<<blocksPerGrid, threadsPerBlock>>>(pos, instances);
}

void CUExecuteKernel(void *devPtr, size_t instances)
{
	//static double dt = 
	particleMovement((float*)devPtr, instances);
}


