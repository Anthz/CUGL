#include "cubufferTab.h"
#include "glsettings.h"


CUBufferTab::CUBufferTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QVBoxLayout;

	bufferLayout = new QHBoxLayout;
	bufferLabel = new QLabel("Buffers:");
	bufferBox = new QComboBox;

	for(int i = 0; i < GLSettings::BufferList.size(); ++i)
	{
		bufferBox->addItem(GLSettings::BufferList.at(i)->bName);
	}

	bufferLayout->addWidget(bufferLabel);
	bufferLayout->addWidget(bufferBox);

	//mainLayout->addWidget(bufferLabel);
	//mainLayout->addWidget(bufferBox);
	mainLayout->addLayout(bufferLayout);

	settingsLayout = new QHBoxLayout;
	
	activatedLabel = new QLabel("Activated:");
	activateBox = new QCheckBox;

	paramIDLabel = new QLabel("Parameter ID:");
	paramIDBox = new QSpinBox;
	paramIDBox->setMinimum(0);
	paramIDBox->setMaximum(9001);

	settingsLayout->addWidget(activatedLabel);
	settingsLayout->addWidget(activateBox);
	settingsLayout->addWidget(paramIDLabel);
	settingsLayout->addWidget(paramIDBox);

	mainLayout->addLayout(settingsLayout);
	mainLayout->setSizeConstraint(QLayout::SetFixedSize);

	setLayout(mainLayout);
}


CUBufferTab::~CUBufferTab()
{
}
