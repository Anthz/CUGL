#include "Utilities.h"

float CUTimer::ElapsedTime()
{
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	Logger::Log("Kernel Time: " + std::to_string(ms) + "ms");
	return ms;
}

namespace Logger
{
	std::string path = "";
	std::string logName = "";
	std::string filePath = "";
	std::string date = "";
	std::string currentTime = "";
	std::ofstream fileStream;

	time_t now;
	tm localTime;

	bool initialised = false;

	void InitLogger(const std::string &folderPath)
	{
		if(!initialised)
		{
			now = time(0);
			localtime_s(&localTime, &now);
			date = std::to_string(localTime.tm_mday) + "-" + std::to_string(localTime.tm_mon + 1) + "-" + std::to_string(localTime.tm_year + 1900);
			currentTime = std::to_string(localTime.tm_hour) + ":" + std::to_string(localTime.tm_min) + ":" + std::to_string(localTime.tm_sec);

			path = folderPath;
			logName = "Log - " + date + ".txt";	//remove /n for filename
			filePath = path + "\\" + logName;
			
			if(CreateDirectoryA(path.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				fileStream.open(filePath, std::ofstream::in);	//check if it exists
				if(fileStream.is_open())
				{
					fileStream.close();
					fileStream.open(filePath, std::ofstream::out | std::ofstream::app);	//if exists, open for append
					fileStream << "\n\n";
				}
				else
				{
					fileStream.close();
					fileStream.open(filePath);	//if it doesn't, create file with read/write permissions
				}

				if(fileStream.is_open())
				{
					fileStream << "[" << currentTime << "]" << " ###LOG START###" << "\n";
					fileStream.close();
				}

				initialised = true;
			}
			else
			{
				fprintf(stderr, "Failed to find/create folder");
			}
		}
	}

	void Log(const std::string &msg)
	{
		if(initialised)
		{
			fileStream.open(filePath, std::ofstream::out | std::ofstream::app);
			if(fileStream.is_open())
			{
				now = time(0);
				localtime_s(&localTime, &now);
				currentTime = std::to_string(localTime.tm_hour) + ":" + std::to_string(localTime.tm_min) + ":" + std::to_string(localTime.tm_sec);
				fileStream << "[" << currentTime << "]" << " " << msg << "\n";
				fileStream.close();
			}
		}
		else
		{
			printf(msg.c_str());
		}
	}
}