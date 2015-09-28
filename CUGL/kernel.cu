#include <Windows.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "Kernel.h"

//normal .cu file with int main() etc
//also with member functions to call kernels
//kernel::doKernel(float* data)
//{
//	kernel<<<grid, block>>>(data)
//}

__global__ void Setup_Rand(curandState *state)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(1234, gid, 0, &state[gid]);	//change seed
}

__global__ void RandomKernel(float *buffer, dim3 dimensions)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= dimensions.x || y >= dimensions.y)
	{
		return;
	}

	buffer[x] *= 1.05f;
}

// int main()
// {
// 	
// }

void Kernel::ExecuteKernel(float *buffer, dim3 dimensions)
{
	dim3 blockDim(18, 1, 1);
	//dim3 gridDim(ceil((float)dimensions.x / (float)blockDim.x), ceil((float)dimensions.y / (float)blockDim.y), 1);

	RandomKernel<<<1, blockDim >>>(buffer, dimensions);
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess)
	{
		printf("Error");
	}

}