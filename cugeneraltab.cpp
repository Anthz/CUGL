#include "cugeneraltab.h"

CUGeneralTab::CUGeneralTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QGridLayout;

	widthLabel = new QLabel("Width:");
	heightLabel = new QLabel("Height:");
	drawLabel = new QLabel("Draw Mode:");
	colourLabel = new QLabel("Clear Colour:");
	vsyncLabel = new QLabel("VSync:");
	msaaLabel = new QLabel("MSAA:");
	fovLabel = new QLabel("FOV:");

	widthBox = new QSpinBox;
	widthBox->setMaximum(1920);
	widthBox->setKeyboardTracking(false);
	//SIGNAL/SLOT if there's parameters
	connect(widthBox, SIGNAL(valueChanged(int)), this, SLOT(WidthChanged(int)));

	heightBox = new QSpinBox;
	heightBox->setMaximum(1024);
	heightBox->setKeyboardTracking(false);
	//connect(heightBox, SIGNAL(valueChanged(int)), this, SLOT(HeightChanged(int)));

	fovBox = new QSpinBox;
	fovBox->setMaximum(120);
	fovBox->setKeyboardTracking(false);
	//connect(fovBox, SIGNAL(valueChanged(int)), this, SLOT(FOVChanged(int)));

	drawBox = new QComboBox;
	drawBox->addItem("GL_TRIANGLES");
	drawBox->addItem("GL_TRIANGLE_STRIP");
	drawBox->addItem("GL_POINTS");
	//connect(drawBox, SIGNAL(currentIndexChanged(int)), this, SLOT(DrawModeChanged(int)));

	colourBox = new ColourTextBox("#6495ED");

	vsyncBox = new QCheckBox;
	vsyncBox->setChecked(true);
	//connect(vsyncBox, SIGNAL(clicked(bool)), this, SLOT(VSyncChanged(bool)));

	msaaBox = new QCheckBox;
	//connect(msaaBox, SIGNAL(clicked(bool)), this, SLOT(MSAAChanged(bool)));

	mainLayout->addWidget(widthLabel, 0, 0);
	mainLayout->addWidget(widthBox, 0, 1);
	mainLayout->addWidget(heightLabel, 1, 0);
	mainLayout->addWidget(heightBox, 1, 1);
	mainLayout->addWidget(drawLabel, 2, 0);
	mainLayout->addWidget(drawBox, 2, 1);
	mainLayout->addWidget(colourLabel, 3, 0);
	mainLayout->addWidget(colourBox, 3, 1);
	mainLayout->addWidget(vsyncLabel, 4, 0);
	mainLayout->addWidget(vsyncBox, 4, 1);
	mainLayout->addWidget(msaaLabel, 5, 0);
	mainLayout->addWidget(msaaBox, 5, 1);
	mainLayout->addWidget(fovLabel, 6, 0);
	mainLayout->addWidget(fovBox, 6, 1);

	setLayout(mainLayout);
}

CUGeneralTab::~CUGeneralTab()
{
	delete mainLayout;
	delete widthLabel;
	delete heightLabel;
	delete drawLabel;
	delete colourLabel;
	delete vsyncLabel; 
	delete msaaLabel;
	delete fovLabel;
	delete widthBox;
	delete heightBox;
	delete fovBox;
	delete drawBox;
	delete colourBox;
	delete vsyncBox;
	delete msaaBox;
}

void CUGeneralTab::WidthChanged(int i)
{
	QMessageBox::information(this, "Changed", "Width Changed");
	//GLWidget::Resize(i, 600);
}

