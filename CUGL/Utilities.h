#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
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
}

static void ErrCheck(cudaError_t e, const char *file, int line)
{
	if(e != cudaSuccess)
	{
		std::stringstream ss;
		ss << "Error: " << file << "|" << line << " " << cudaGetErrorString(e);
		Logger::Log(ss.str());
	}
}

class CUTimer
{
public:
	CUTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
	~CUTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

	inline void Begin() { cudaEventRecord(start); }
	inline void End() { cudaEventRecord(stop); cudaEventSynchronize(stop); }
	inline void ElapsedTime(float &ms) { cudaEventElapsedTime(&ms, start, stop); }

private:
	cudaEvent_t start, stop;
};
