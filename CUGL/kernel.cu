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

__global__ void RandomKernel(cudaGraphicsResource *buffer, dim3 dimensions)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= dimensions.x || y >= dimensions.y)
	{
		return;
	}

	float4 element = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
	//surf2Dwrite(element, tex, x * sizeof(float4), y);	//undefined in .cu file
}

// int main()
// {
// 	
// }

void Kernel::ExecuteKernel(cudaGraphicsResource *buffer, dim3 dimensions)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim(ceil((float)dimensions.x / (float)blockDim.x), ceil((float)dimensions.y / (float)blockDim.y), 1);

	RandomKernel << <gridDim, blockDim >> >(buffer, dimensions);
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess)
	{
		printf("Error");
	}

}