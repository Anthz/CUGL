#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <string>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>

#define ERRORCHECK(e) (ErrCheck(e, __FILE__, __LINE__))

namespace Logger
{
	void InitLogger(const std::string &folderPath = "Logs");
	void Log(const std::string &msg);
	void BasicLog(const std::string &msg, std::string path);
}

static bool ErrCheck(cudaError_t e, const char *file, int line)
{
	if(e != cudaSuccess)
	{
		std::stringstream ss;
		ss << "Error: " << file << "|" << line << " " << cudaGetErrorString(e);
		Logger::Log(ss.str());
		return false;
	}
	else
	{
		return true;
	}
}

class CUTimer
{
public:
	CUTimer(std::string path) { filePath = path; cudaEventCreate(&start); cudaEventCreate(&stop); }
	~CUTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

	inline void Begin() { cudaEventRecord(start); }
	inline void End() { cudaEventRecord(stop); cudaEventSynchronize(stop); ElapsedTime(); }
	float ElapsedTime();

private:
	cudaEvent_t start, stop;
	std::string filePath;
};

class Timer
{
public:
	Timer(std::string path) { filePath = path; QueryPerformanceFrequency(&frequency); }
	~Timer();

	inline void Begin() { QueryPerformanceCounter(&t1); }
	inline void End() { QueryPerformanceCounter(&t2); ElapsedTime(); }
	double ElapsedTime();

private:
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	double elapsedTime;
	std::string filePath;
};

namespace CUMath
{
	class SimpleComplex
	{
	public:
		__host__ __device__ SimpleComplex();
		__host__ __device__ SimpleComplex(float real, float imag);

		__host__ __device__ static SimpleComplex Polar(float amp, float theta);

		__host__ __device__ float abs();
		__host__ __device__ float absSq();
		__host__ __device__ float Angle();
		__host__ __device__ void Normalise();
		__host__ __device__ int Measure();

		__host__ __device__ SimpleComplex operator*(const float& f) const;
		__host__ __device__ SimpleComplex operator*(const int& i) const;
		__host__ __device__ SimpleComplex operator/(const float& f) const;
		__host__ __device__ SimpleComplex operator+(const SimpleComplex& c) const;
	private:
		float real, imag;
	};
}
