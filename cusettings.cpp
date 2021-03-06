#include "cusettings.h"

std::vector<CUGLBuffer*> CUSettings::BufferList;

//param > buffer
//param - select from objects properties, buffers - order

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
	connect(tabs, SIGNAL(currentChanged(int)), this, SLOT(TabChanged(int)));

	settingsLayout->addWidget(tabs);
	settingsGroup->setLayout(settingsLayout);
	mainLayout->addWidget(settingsGroup);

	setLayout(mainLayout);

	CUInit();
}

CUSettings::~CUSettings()
{
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

	static_cast<CUGeneralTab*>(tabs->widget(0))->UpdateDevices(deviceList);
}

void CUSettings::TabChanged(int i)
{
	switch(i)
	{
	case 0:
		//CUGeneral
		break;
	case 1:
		//CUParams
		break;
	case 2:
		//CUBuffers
		static_cast<CUBufferTab*>(tabs->currentWidget())->Update();
		break;
	case 3:
		//CUKernel
		break;
	}
}

QSize CUSettings::minimumSizeHint() const
{
	return QSize(500, 350);
}

QSize CUSettings::sizeHint() const
{
	return QSize((int)static_cast<QWidget*>(parent())->width() / 3, 350);
}