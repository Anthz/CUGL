#include "cugeneraltab.h"

CUGeneralTab::CUGeneralTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QGridLayout;

	deviceLabel = new QLabel("Device:");

	deviceBox = new QComboBox;
	connect(deviceBox, SIGNAL(currentIndexChanged(int)), this, SLOT(DeviceChanged(int)));

	mainLayout->addWidget(deviceLabel, 0, 0);
	mainLayout->addWidget(deviceBox, 0, 1);

	setLayout(mainLayout);
}

CUGeneralTab::~CUGeneralTab()
{
 	delete mainLayout;
}

void CUGeneralTab::DeviceChanged(int i)
{
	//QMessageBox::information(this, "Changed", "Width Changed");
	cudaSetDevice(i);
}

void CUGeneralTab::UpdateDevices(std::vector<Device>& dList)
{
	for each (Device d in dList)
	{
		deviceBox->addItem(d.name);
	}
}

