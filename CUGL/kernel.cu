#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//normal .cu file with int main() etc
//also with member functions to call kernels
//kernel::doKernel(float* data)
//{
//	kernel<<<grid, block>>>(data)
//}