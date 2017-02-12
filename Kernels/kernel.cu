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
#include <time.h>
#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "math_constants.h"
#include "helper_math.h"
#include "utilities.h"

//non-pow2 PBO doesn't work
#define N_X 256	//not inc ghost
#define N_Y 256	//not inc ghost
#define N N_X * N_Y	//not inc ghost
#define BLOCK_SIZE_X 16 //inc ghost/overlap	//large size should hide wasted threads better - CHECK!
#define BLOCK_SIZE_Y 16 //inc ghost/overlap
#define BLOCK_SIZE_COPY_X 16
#define BLOCK_SIZE_COPY_Y 16
#define N_BLOCKS_X (N_X + (BLOCK_SIZE_X - 2) - 1) / (BLOCK_SIZE_X - 2) //-2 for increased no blocks for ghost cells
#define N_BLOCKS_Y (N_Y + (BLOCK_SIZE_Y - 2) - 1) / (BLOCK_SIZE_Y - 2)
#define N_BLOCKS_RAND (N + (BLOCK_SIZE_X * BLOCK_SIZE_Y) - 1) / (BLOCK_SIZE_X * BLOCK_SIZE_Y)
#define N_BLOCKS_COPY_X (N_X + BLOCK_SIZE_COPY_X - 1) / BLOCK_SIZE_COPY_X
#define N_BLOCKS_COPY_Y (N_Y + BLOCK_SIZE_COPY_Y - 1) / BLOCK_SIZE_COPY_Y
#define STREAM_COUNT 5

namespace CUMath
{
	//non-empty constructor was drastically slowing down performance (1->10ms)
	__host__ __device__ SimpleComplex::SimpleComplex()
	{
		//real = 0.0f;
		//imag = 0.0f;
	}

	__host__ __device__ SimpleComplex::SimpleComplex(float real, float imag)
	{
		this->real = real;
		this->imag = imag;
	}

	__host__ __device__ SimpleComplex SimpleComplex::Polar(float amp, float theta)
	{
		return SimpleComplex(cosf(theta) * amp, sinf(theta) * amp);
	}

	__host__ __device__ float SimpleComplex::abs()
	{
		return sqrt(real * real + imag * imag);
	}
	__host__ __device__ float SimpleComplex::absSq()
	{
		return real * real + imag * imag;
	}

	__host__ __device__ float SimpleComplex::Angle()
	{
		float angle = 0.0f;
		angle = atan2f(imag, real);

		if(real < 0)
		{
			angle += CUDART_PI_F;
		}
		else if(angle < 0)
		{
			angle += CUDART_PI_F * 2.0f;
		}

		return angle;
	}

	__host__ __device__ SimpleComplex SimpleComplex::operator*(const float& f) const
	{
		return SimpleComplex(real * f, imag * f);
	}

	__host__ __device__ SimpleComplex SimpleComplex::operator/(const float& f) const
	{
		return SimpleComplex(real / f, imag / f);
	}

	__host__ __device__ SimpleComplex SimpleComplex::operator+(const SimpleComplex& c) const
	{
		return SimpleComplex(real + c.real, imag + c.imag);
	}
}

struct HQubit
{
	std::complex<float> zero = std::complex<float>();
	std::complex<float> one = std::complex<float>();

	HQubit(){};

	HQubit(float amp, float theta)
	{
		one = std::polar<float>(amp, theta); //std::complex<float>(amp, theta);
		zero = std::polar<float>(1 - amp, theta);
	}

	//QCA
	//|modulus| of sum of surrounding neighbours = rule to apply

	void Normalise()
	{
		double norm = sqrt(abs(zero) * abs(zero) + abs(one) * abs(one));	//sqrt(pow(abs(zero), 2) + pow(abs(one), 2));
		zero /= norm;
		one /= norm;
	}

	int Measure()
	{
		float oneProb = abs(one) * abs(one);	//pow(abs(one), 2);
		float roll = (rand() / (float)RAND_MAX);

		if(roll < oneProb)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
};

struct DQubit
{
	CUMath::SimpleComplex zero, one;

	__host__ __device__ DQubit(){}

	__host__ __device__ DQubit(float amp, float theta)
	{
		one = CUMath::SimpleComplex::Polar(amp, theta);
		zero = CUMath::SimpleComplex::Polar(sqrtf(1.0f - (amp * amp)), theta);
	}

	//QCA
	//|modulus| of sum of surrounding neighbours = rule to apply
	//sum of phase = theta to use in OP

	__host__ __device__ void Normalise()
	{
		float norm = sqrtf(zero.absSq() + one.absSq());	//sqrt(pow(abs(zero), 2) + pow(thrust::abs(one), 2));
		one = one / norm;
		zero = zero / norm;
	}

	//to-do: implement prob field with white = dead % and varying colours (from angle) = alive
	//look up gradient implementation
	__device__ int Measure(curandState *state)
	{
		//prob of finding qubit in |1>
		float oneProb = one.absSq();

		//curand 0-1
		if(curand_uniform(state) < oneProb)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
};

__device__ curandState randArray[N];	//rand doesn't need extra

static curandState* randPtr;
static int *lifePtr;
static int *lifeTempPtr;
static DQubit *qubitPtr;
static DQubit *qubitTempPtr;
static int *cornersPtr;
static CUTimer t("Timings\\QGoL_Shared_256_5.txt");
static cudaStream_t streams[STREAM_COUNT];

__global__ void setup_rand(curandState *state)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	if(offset < (N_X * N_Y))
	{
		curand_init((unsigned long long)clock(), offset, 0, &state[offset]); //random seed of (unsigned long long)clock()
	}
}

__global__ void CopyToTop(int *life, DQubit *qubits)
{
	//copy bottom row to top (ghost cells)
	int x = threadIdx.x + blockIdx.x * blockDim.x + 1;	//threads = 1 <= N_X

	if(x <= N_X)
	{
		life[x] = life[x + N_Y * (N_X + 2)];	//N_Y
		qubits[x] = qubits[x + N_Y * (N_X + 2)];
	}
}

__global__ void CopyToBottom(int *life, DQubit *qubits)
{
	//copy top row to bottom (ghost cells)
	int x = threadIdx.x + blockIdx.x * blockDim.x + 1;	//threads = 1 <= N_X

	if(x <= N_X)
	{
		life[x + (N_Y + 1) * (N_X + 2)] = life[x + N_X + 2];
		qubits[x + (N_Y + 1) * (N_X + 2)] = qubits[x + N_X + 2];
	}
}

__global__ void CopyToLeft(int *life, DQubit *qubits)
{
	//copy right column to left (ghost cells)
	int x = threadIdx.x + blockIdx.x * blockDim.x + 1;	//threads = 1 <= N_Y

	if(x <= N_Y)
	{
		x *= N_X + 2;	//transform to vertical

		life[x] = life[x + N_X];
		qubits[x] = qubits[x + N_X];
	}
}

__global__ void CopyToRight(int *life, DQubit *qubits)
{
	//copy left column to right (ghost cells)
	int x = threadIdx.x + blockIdx.x * blockDim.x + 1;	//threads = 1 <= N_Y

	if(x <= N_Y)
	{
		x *= N_X + 2;	//transform to vertical

		life[x + (N_X + 1)] = life[x + 1];
		qubits[x + (N_X + 1)] = qubits[x + 1];
	}
}

//ids contains destination and source ids (0-3 = dest, 4-7 = source)
__global__ void CopyToCorners(int *life, DQubit *qubits, int ids[])
{
	//copy left column to right (ghost cells)
	int x = threadIdx.x + blockIdx.x * blockDim.x;	//threads = 0 < 4
	life[ids[x]] = life[ids[x + 4]];
	qubits[ids[x]] = qubits[ids[x + 4]];
}

inline
__device__ void BirthOp(DQubit *q, float thetaSum, float multiplier = 1.0f)
{
	//a+|b|e^(i*theta)

	//*q = (*q + DQubit::Polar(q->ZeroStateAmp, thetaSum)) * multiplier;
	q->one = (q->one + CUMath::SimpleComplex::Polar(q->zero.abs(), thetaSum)) * multiplier;
	q->zero = CUMath::SimpleComplex(0.0f, 0.0f);
}

//|0> angle does not effect simulations?
inline
__device__ void DeathOp(DQubit *q, float thetaSum, float multiplier = 1.0f)
{
	//b + |a|e^(i*theta)

	//*q = (DQubit::Polar(q->ZeroStateAmp, q->Angle()) + DQubit::Polar(q->cAbs(), thetaSum)) * multiplier;
	q->zero = (q->zero + CUMath::SimpleComplex::Polar(q->one.abs(), thetaSum)) * multiplier;
	q->one = CUMath::SimpleComplex(0.0f, 0.0f);
}

//optimise qubits passed in
//load in block to shared
//watch out for overlap (load extra or use global sparingly)
inline
__device__ CUMath::SimpleComplex SumNeighbours(DQubit(*qubits)[BLOCK_SIZE_X][BLOCK_SIZE_Y], int(*lifeData)[BLOCK_SIZE_X][BLOCK_SIZE_Y], int x, int y)
{
	//sum of all surrounding superpositions (Ae^(i*theta))

	CUMath::SimpleComplex sum = CUMath::SimpleComplex(0.0f, 0.0f);

	if((*lifeData)[x - 1][y - 1] == 1)
		sum = sum + (*qubits)[x - 1][y - 1].one;
	if((*lifeData)[x][y - 1] == 1)
		sum = sum + (*qubits)[x][y - 1].one;
	if((*lifeData)[x + 1][y - 1] == 1)
		sum = sum + (*qubits)[x + 1][y - 1].one;
	if((*lifeData)[x - 1][y] == 1)
		sum = sum + (*qubits)[x - 1][y].one;
	if((*lifeData)[x + 1][y] == 1)
		sum = sum + (*qubits)[x + 1][y].one;
	if((*lifeData)[x - 1][y + 1] == 1)
		sum = sum + (*qubits)[x - 1][y + 1].one;
	if((*lifeData)[x][y + 1] == 1)
		sum = sum + (*qubits)[x][y + 1].one;
	if((*lifeData)[x + 1][y + 1] == 1)
		sum = sum + (*qubits)[x + 1][y + 1].one;

	//no if statements - causes NaN errors with NSIGHT (probably from casting 0 to float)
	/*sum += (*qubits)[x - 1][y - 1].one * (float)(*lifeData)[x - 1][y - 1];
	sum += (*qubits)[x][y - 1].one * (float)(*lifeData)[x][y - 1];
	sum += (*qubits)[x + 1][y - 1].one * (float)(*lifeData)[x + 1][y - 1];
	sum += (*qubits)[x - 1][y].one * (float)(*lifeData)[x - 1][y];
	sum += (*qubits)[x + 1][y].one * (float)(*lifeData)[x + 1][y];
	sum += (*qubits)[x - 1][y + 1].one * (float)(*lifeData)[x - 1][y + 1];
	sum += (*qubits)[x][y + 1].one * (float)(*lifeData)[x][y + 1];
	sum += (*qubits)[x + 1][y + 1].one * (float)(*lifeData)[x + 1][y + 1];*/

	return sum;
}

//remove temps from params
__global__
void QGOL(float *topProb, float *bottomProb, int *lifeBuf, int *output, DQubit *qubits, DQubit *tempQubits, curandState *states)
{
	int t_X = threadIdx.x;
	int t_Y = threadIdx.y;
	//new x/y coords with ghost cells
	int x = t_X + blockIdx.x * (blockDim.x - 2);	//pushback id to cover overlap column from prev block
	int y = t_Y + blockIdx.y * (blockDim.y - 2);	//pushback id to cover overlap row from prev block
	int offset = x + y * (N_X + 2);	//use N_X over blockDim * gridDim due to overlapping blocks
	int innerOffset = (x - 1) + (y - 1) * N_X;	//move up/left diag and put into inner grid dim

	//change to 1D array and test perf
	__shared__ int s_Life[BLOCK_SIZE_X][BLOCK_SIZE_Y];
	__shared__ DQubit s_Qubits[BLOCK_SIZE_X][BLOCK_SIZE_Y];

	//if in range (inc ghost cells)
	if(x < (N_X + 2) && y < (N_Y + 2))
	{
		s_Life[t_X][t_Y] = lifeBuf[offset];
		s_Qubits[t_X][t_Y] = qubits[offset];
	}

	__syncthreads();

	if(x > 0 && y > 0 && x < (N_X + 1) && y < (N_Y + 1))	//skips ghost cells
	{
		if(t_X > 0 && t_X < blockDim.x - 1 && t_Y > 0 && t_Y < blockDim.y - 1)	//skip overlap cells
		{
			//sum neighbours
			//take abs(sum) for neighbour count
			//take angle from sum to use in various ops
			//perform op
			//normalise

			curandState localState = states[innerOffset];	//use inner grid id to reduce number of rand variables
			DQubit q = s_Qubits[t_X][t_Y];
			DQubit qtemp = q;

			CUMath::SimpleComplex sum = SumNeighbours(&s_Qubits, &s_Life, t_X, t_Y);
			float mod = sum.abs();
			float sumAngle = sum.Angle();

			//determine op from mod
			//0 <= A <= 1 op = D
			//1 < A <= 2 op = (sqrt(2)+1)(2-A)D + (A-1)S
			//2 < A <= 3 op = (sqrt(2)+1)(3-A)S + (A-2)B
			//3 < A < 4 op = (sqrt(2)+1)(4-A)B + (A-3)D
			//A >= 4 op = D

			if(mod >= 0 && mod <= 1)
			{
				//D
				DeathOp(&q, sumAngle);	//.imag != angle	//change 1st/last OP to q = 0 for optimisation
			}
			else if(mod > 1 && mod <= 2)
			{
				//D & S
				DeathOp(&q, sumAngle, (sqrtf(2.0f) + 1.0f) * (2.0f - mod));
				qtemp.zero = qtemp.zero * (mod - 1.0f);
				qtemp.one = qtemp.one * (mod - 1.0f);
				q.zero = q.zero + qtemp.zero;
				q.one = q.one + qtemp.one;
			}
			else if(mod > 2 && mod <= 3)
			{
				//S & B
				q.zero = q.zero * (sqrtf(2.0f) + 1.0f) * (3.0f - mod);
				q.one = q.one * (sqrtf(2.0f) + 1.0f) * (3.0f - mod);
				BirthOp(&qtemp, sumAngle, (mod - 2.0f));
				q.zero = q.zero + qtemp.zero;
				q.one = q.one + qtemp.one;
			}
			else if(mod > 3 && mod < 4)
			{
				//B & D
				BirthOp(&q, sumAngle, (sqrtf(2.0f) + 1.0f) * (4.0f - mod));
				DeathOp(&qtemp, sumAngle, (mod - 3.0f));
				q.zero = q.zero + qtemp.zero;
				q.one = q.one + qtemp.one;
			}
			else if(mod >= 4)
			{
				//D
				DeathOp(&q, sumAngle);
			}
			else
			{
				//something went wrong
				assert(0);
			}

			//measure state to update lifebuf
			//update tex

			q.Normalise();

			int alive = q.Measure(&localState);

			output[offset] = alive;
			topProb[innerOffset] = q.one.absSq();
			bottomProb[innerOffset] = q.zero.absSq();
			//texBuf[innerOffset].x = (alive == 0) ? 255 : 0;	//check perf
			//texBuf[offset].y = (abs(qtemp2.one) == 0) ? 255 : 0;	//for multi-colour cells (D/D = White|D/A = Green|A/D = Purple|A/A = Black)
			//texBuf[offset].y = (alive == 0) ? 255 : 0;
			//texBuf[offset].z = (alive == 0) ? 255 : 0;
			//texBuf[offset].w = 255;
			tempQubits[offset] = q;
		}
	}
}

void FindCornerIDs(int *ids)
{
	ids[0] = 0;	//top left
	ids[1] = N_X + 1;	//top right
	ids[2] = (N_X + 2) * (N_Y + 2) - (N_X + 2);	//bottom left
	ids[3] = (N_X + 2) * (N_Y + 2) - 1;	//bottom right
	ids[4] = ids[3] - (N_X + 2) - 1;
	ids[5] = ids[4] - (N_X - 1);
	ids[6] = (N_X + 2) + 1 + (N_X - 1);
	ids[7] = ids[6] - (N_X - 1);
}

void CUSetup()
{
	//cudaOccupancyMaxPotentialBlockSize(...
	for(int i = 0; i < STREAM_COUNT; i++)
	{
		cudaStreamCreate(&streams[i]);
	}

	ERRORCHECK(cudaGetSymbolAddress((void**)&randPtr, randArray));
	setup_rand << <N_BLOCKS_RAND, dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y) >> >(randPtr);
	ERRORCHECK(cudaGetLastError());

	int *hCornerPtr = (int*)malloc(8 * sizeof(int));
	FindCornerIDs(hCornerPtr);
	ERRORCHECK(cudaMalloc((void**)&cornersPtr, 8 * sizeof(int)));
	ERRORCHECK(cudaMemcpy(cornersPtr, hCornerPtr, 8 * sizeof(int), cudaMemcpyHostToDevice));

	int *hLifePtr = (int*)malloc((N_X + 2) * (N_Y + 2) * sizeof(int));
	DQubit *hQubitPtr = (DQubit*)malloc((N_X + 2) * (N_Y + 2) * sizeof(DQubit));

	for(int i = 0; i < (N_X + 2) * (N_Y + 2); ++i)
	{
		hLifePtr[i] = (1.5f - 0.5f) * (rand() / (float)RAND_MAX) + 0.5f;
		hQubitPtr[i] = DQubit(hLifePtr[i], (rand() / (float)RAND_MAX) * (2 * CUDART_PI_F));
	}
	
	ERRORCHECK(cudaMalloc((void**)&lifePtr, (N_X + 2) * (N_Y + 2) * sizeof(int)));
	ERRORCHECK(cudaMalloc((void**)&lifeTempPtr, (N_X + 2) * (N_Y + 2) * sizeof(int)));
	ERRORCHECK(cudaMemcpy(lifePtr, hLifePtr, (N_X + 2) * (N_Y + 2) * sizeof(int), cudaMemcpyHostToDevice));
	ERRORCHECK(cudaMemcpy(lifeTempPtr, lifePtr, (N_X + 2) * (N_Y + 2) * sizeof(int), cudaMemcpyDeviceToDevice));

	ERRORCHECK(cudaMalloc((void**)&qubitPtr, (N_X + 2) * (N_Y + 2) * sizeof(DQubit)));
	ERRORCHECK(cudaMalloc((void**)&qubitTempPtr, (N_X + 2) * (N_Y + 2) * sizeof(DQubit)));
	ERRORCHECK(cudaMemcpy(qubitPtr, hQubitPtr, (N_X + 2) * (N_Y + 2) * sizeof(DQubit), cudaMemcpyHostToDevice));
	ERRORCHECK(cudaMemcpy(qubitTempPtr, qubitPtr, (N_X + 2) * (N_Y + 2) * sizeof(DQubit), cudaMemcpyDeviceToDevice));

	free(hCornerPtr);
	free(hLifePtr);
	free(hQubitPtr);
}

//UI option for iterations
void CUExecuteKernel(std::vector<void*> *params)	//std::vector<void*> *params, size_t instances, float dt
{
	t.Begin();

	CopyToTop << <N_BLOCKS_COPY_X, BLOCK_SIZE_COPY_X, 0, streams[0] >> >(lifePtr, qubitPtr);
	CopyToBottom << <N_BLOCKS_COPY_X, BLOCK_SIZE_COPY_X, 0, streams[1] >> >(lifePtr, qubitPtr);
	CopyToLeft << <N_BLOCKS_COPY_Y, BLOCK_SIZE_COPY_Y, 0, streams[2] >> >(lifePtr, qubitPtr);
	CopyToRight << <N_BLOCKS_COPY_Y, BLOCK_SIZE_COPY_Y, 0, streams[3] >> >(lifePtr, qubitPtr);
	CopyToCorners << <1, 4, 0, streams[4] >> >(lifePtr, qubitPtr, cornersPtr);

	cudaDeviceSynchronize();

	QGOL << <dim3(N_BLOCKS_X, N_BLOCKS_Y), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y) >> >((float*)(params->at(0)), (float*)(params->at(1)), lifePtr, lifeTempPtr, qubitPtr, qubitTempPtr, randPtr);

	t.End();

	//normal swap since static
	std::swap(lifePtr, lifeTempPtr);
	std::swap(qubitPtr, qubitTempPtr);

	ERRORCHECK(cudaGetLastError());
}