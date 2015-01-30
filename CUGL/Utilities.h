#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <time.h>
#include <iostream>
#include <fstream>
#include <windows.h>

namespace Logger
{
	void InitLogger(const std::string &folderPath = "Logs");
	void Log(const std::string &msg);
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
