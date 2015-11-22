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
#include <vector>
#include "math_constants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_BODIES 30000
#define DT 0.01f // time step
#define N_ITERS 10
#define BLOCK_SIZE 256
#define N_BLOCKS (N_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE
#define SOFTENING 1e-9f

static bool init = false;

void randomizeBodies(float *data, int n) {
	for(int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}

__global__
void bodyForce(float *pos, float *vel) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < N_BODIES)
	{
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for(int j = 0; j < N_BODIES; j++) {
			float dx = pos[j] - pos[i * 3 + 0];
			float dy = pos[j] - pos[i * 3 + 1];
			float dz = pos[j] - pos[i * 3 + 2];
			float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
			float invDist = rsqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;

			Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
		}

		vel[i * 3 + 0] += DT*Fx; vel[i * 3 + 1] += DT*Fy; vel[i * 3 + 2] += DT*Fz;
	}
}

__global__
void integratePos(float* pos, float* vel)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i < N_BODIES)
	{
		pos[i * 3 + 0] += vel[i * 3 + 0] * DT;
		pos[i * 3 + 1] += vel[i * 3 + 1] * DT;
		pos[i * 3 + 2] += vel[i * 3 + 2] * DT;
	}
}

void NBody(float *posBuf, float *velBuf)
{
	bodyForce << <N_BLOCKS, BLOCK_SIZE >> >(posBuf, velBuf); // compute interbody forces
	integratePos << <N_BLOCKS, BLOCK_SIZE >> >(posBuf, velBuf);
} 

//UI option for iterations
void CUExecuteKernel(std::vector<void*> *params)	//std::vector<void*> *params, size_t instances, float dt
{
	NBody((float*)(params->at(0)), (float*)(params->at(1)));
}