#include "cusettings.h"

CUSettings::CUSettings(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QVBoxLayout();
	settingsLayout = new QVBoxLayout();

	settingsGroup = new QGroupBox("CUDA Settings");
	tabs = new QTabWidget();
	tabs->addTab(new CUGeneralTab(), "General");
	tabs->addTab(new ParamTab(), "Parameters");
	tabs->addTab(new CUBufferTab(), "Buffers");
	tabs->addTab(new KernelTab(), "Kernel");
	settingsLayout->addWidget(tabs);
	settingsGroup->setLayout(settingsLayout);
	mainLayout->addWidget(settingsGroup);

	setLayout(mainLayout);

	CUInit();
}

CUSettings::~CUSettings()
{
	delete tabs;
	delete settingsLayout;
	delete settingsGroup;
	delete mainLayout;
}

void CUSettings::CUInit()
{
	deviceCount = 0;

	cuInit(0);
	cuDeviceGetCount(&deviceCount);

	for(int i = 0; i < deviceCount; ++i)
	{
		Device d;
		cuDeviceGet(&d.handle, i);
		cuDeviceGetName(d.name, 100, d.handle);
		deviceList.push_back(d);
		printf("Device %i: %s", d.handle, d.name);
	}

	cudaSetDevice(0);
}

