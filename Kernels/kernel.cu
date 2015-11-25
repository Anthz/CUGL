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

#define N_BODIES 8192
#define DT 0.1f // time step
#define N_ITERS 10
#define BLOCK_SIZE 256
#define N_BLOCKS (N_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE
#define SOFTENING 1e-9f

//G = 6.67300 × 10^−11 m^3/kg s^2

__device__
float4 CalcForce(float4 *pos, float dt, float4 derivative)
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

		float dx = pOther.x - p.x;
		float dy = pOther.y - p.y;
		float dz = pOther.z - p.z;
		float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
		float invDist = rsqrtf(distSqr);
		float invDist3 = invDist * invDist * invDist;
		float s = pOther.w * invDist3;

		Fx += dx * s; Fy += dy * s; Fz += dz * s;
		//Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
	}

	return make_float4(dt*Fx, dt*Fy, dt*Fz, 1.0f);
}

__device__
float4 CalcVel(float4* vel, float dt, float4 derivative)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float4 v = vel[i];
	v.x += derivative.x;
	v.y += derivative.y;
	v.z += derivative.z;
	
	v.x = v.x * dt;
	v.y = v.y * dt;
	v.z = v.z * dt;

	return v;
}

__global__
void NBody(float4 *posBuf, float4 *velBuf, float4 *pTemp, float4 *vTemp)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < N_BODIES)
	{
		float4 v1 = CalcForce(posBuf, DT * 0.0, make_float4(0.0f, 0.0f, 0.0f, 0.0f));	//float4()
		float4 p1 = CalcVel(velBuf, DT * 0.0, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
		
		float4 v2 = CalcForce(posBuf, DT * 0.5, (DT / 2 * p1));
		float4 p2 = CalcVel(velBuf, DT * 0.5, (DT / 2 * v1));

		float4 v3 = CalcForce(posBuf, DT * 0.5, (DT / 2 * p2));
		float4 p3 = CalcVel(velBuf, DT * 0.5, (DT / 2 * v2));

		float4 v4 = CalcForce(posBuf, DT * 1.0, (DT * p3));		
		float4 p4 = CalcVel(velBuf, DT * 1.0, (DT * v3));

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