#include "cubufferTab.h"
#include "glsettings.h"
#include "cusettings.h"

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
	connect(bufferBox, SIGNAL(currentIndexChanged(int)), this, SLOT(BufferChanged(int)));

	bufferLayout->addWidget(bufferLabel);
	bufferLayout->addWidget(bufferBox);

	mainLayout->addLayout(bufferLayout);

	settingsLayout = new QHBoxLayout;

	activatedLabel = new QLabel("Activated:");
	activateBox = new QCheckBox;
	connect(activateBox, SIGNAL(stateChanged(int)), this, SLOT(ActivatedChanged(int)));

	paramIDLabel = new QLabel("Parameter ID:");
	paramIDBox = new QSpinBox;
	paramIDBox->setMinimum(-1);
	paramIDBox->setMaximum(9001);
	connect(paramIDBox, SIGNAL(valueChanged(int)), this, SLOT(ParamIDChanged(int)));

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
	delete mainLayout;
}

void CUBufferTab::Update()
{
	bufferBox->clear();

	for(int i = 0; i < GLSettings::BufferList.size(); ++i)
	{
		bufferBox->addItem(GLSettings::BufferList.at(i)->bName);
	}

	//doesn't always work due to removing buffers
	/*for(int i = 0; i < GLSettings::BufferList.size(); ++i)
	{
	if(bufferBox->findText(GLSettings::BufferList.at(i)->bName) == -1)
	bufferBox->addItem(GLSettings::BufferList.at(i)->bName);
	}*/
}

void CUBufferTab::BufferChanged(int i)
{
	if(i != -1)
	{
		activateBox->setChecked(GLSettings::BufferList.at(i)->Cuda());
		paramIDBox->setValue(GLSettings::BufferList.at(i)->ParamID());
	}
}

void CUBufferTab::ActivatedChanged(int i)
{
	CUGLBuffer *b = GLSettings::BufferList.at(bufferBox->currentIndex());
	b->Cuda((i == 0) ? false : true);
	CUSettings::BufferList.push_back(b);
}

void CUBufferTab::ParamIDChanged(int i)
{
	GLSettings::BufferList.at(bufferBox->currentIndex())->ParamID(i);
}
