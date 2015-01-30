#pragma once
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

