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
#include "helper_math.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"

#define N_BODIES 601
#define DT 5000.0f // time step
#define N_ITERS 10
#define BLOCK_SIZE 256
#define N_BLOCKS (N_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE
#define SOFTENING 1e-9f
#define G 6.67e-11 //4.0 * 3.14159265 * 3.14159265	//6.67e-11

__device__
float4 CalcForce(float4 *pos, float4 derivative)
{
	//calc or pass as param?
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float4 p = pos[i];
	p.x += derivative.x;
	p.y += derivative.y;
	p.z += derivative.z;

	float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

	//#pragma unroll - slower with unroll?
	for(int j = 0; j < N_BODIES; j++)
	{
		if(i == j)
			continue;

		float4 pOther = pos[j];

		float dx = p.x - pOther.x;	//pOther.x - p.x;
		float dy = p.y - pOther.y;	//pOther.y - p.y;
		float dz = p.z - pOther.z;	//pOther.z - p.z;
		float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
		float dist = sqrtf(distSqr);
		//float invDist = rsqrtf(distSqr);
		//float invDist3 = invDist * invDist * invDist;
		float s = -(G * pOther.w) / (dist * dist); //pOther.w * invDist3;

		Fx += (s / dist) * dx; Fy += (s / dist) * dy; Fz += (s / dist) * dz;
		//Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
	}

	return make_float4(Fx, Fy, Fz, 1.0f);
}

__device__
float4 CalcVel(float4* vel, float4 derivative)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float4 v = vel[i];
	v.x += derivative.x;
	v.y += derivative.y;
	v.z += derivative.z;

	return v;
}

__global__
void NBody(float4 *posBuf, float4 *velBuf, float4 *pTemp, float4 *vTemp)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < N_BODIES)
	{
		float4 v1 = CalcForce(posBuf, make_float4(0.0f, 0.0f, 0.0f, 0.0f));	//float4()
		float4 p1 = velBuf[i]; //CalcVel(velBuf, make_float4(0.0f, 0.0f, 0.0f, 0.0f));

		float4 v2 = CalcForce(posBuf, (DT / 2 * p1));
		float4 p2 = CalcVel(velBuf, (DT / 2 * v1));

		float4 v3 = CalcForce(posBuf, (DT / 2 * p2));
		float4 p3 = CalcVel(velBuf, (DT / 2 * v2));

		float4 v4 = CalcForce(posBuf, (DT * p3));
		float4 p4 = CalcVel(velBuf, (DT * v3));

		vTemp[i] = (DT ⁄ 6) * (v1 + (2.0f * v2) + (2.0f * v3) + v4);
		pTemp[i] = (DT ⁄ 6) * (p1 + (2.0f * p2) + (2.0f * p3) + p4);
	}
}

__global__
void Update(float4 *posBuf, float4 *velBuf, float4 *pTemp, float4 *vTemp)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < N_BODIES)
	{
		float4 vel = velBuf[i];
		float4 velT = vTemp[i];
		float4 pos = posBuf[i];
		float4 posT = pTemp[i];

		vel.x += velT.x;
		vel.y += velT.y;
		vel.z += velT.z;

		pos.x += posT.x;
		pos.y += posT.y;
		pos.z += posT.z;

		velBuf[i] = vel;
		posBuf[i] = pos;
	}
}

//UI option for iterations
void CUExecuteKernel(std::vector<void*> *params)	//std::vector<void*> *params, size_t instances, float dt
{
	//CUTimer t;
	float4 *pTemp = 0;
	float4 *vTemp = 0;
	ERRORCHECK(cudaMalloc((void**)&pTemp, sizeof(float4) * N_BODIES));
	ERRORCHECK(cudaMalloc((void**)&vTemp, sizeof(float4) * N_BODIES));

	//t.Begin();
	NBody << <N_BLOCKS, BLOCK_SIZE >> >((float4*)(params->at(0)), (float4*)(params->at(1)), pTemp, vTemp);
	Update << <N_BLOCKS, BLOCK_SIZE >> >((float4*)(params->at(0)), (float4*)(params->at(1)), pTemp, vTemp);
	//t.End();

	ERRORCHECK(cudaFree(pTemp));
	ERRORCHECK(cudaFree(vTemp));

}