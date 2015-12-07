#include "glgeneraltab.h"

GLGeneralTab::GLGeneralTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QGridLayout;

	widthLabel = new QLabel("Width:");
	heightLabel = new QLabel("Height:");
	projMatrixLabel = new QLabel("Projection:");
	fovLabel = new QLabel("VFOV:");
	aspectLabel = new QLabel("Aspect:");
	nearLabel = new QLabel("Near:");
	farLabel = new QLabel("Far:");
	leftLabel = new QLabel("Left:");
	rightLabel = new QLabel("Right:");
	bottomLabel = new QLabel("Bottom");
	topLabel = new QLabel("Top:");
	drawLabel = new QLabel("Draw Mode:");
	colourLabel = new QLabel("Clear Colour:");
	vsyncLabel = new QLabel("VSync:");
	msaaLabel = new QLabel("MSAA:");

	widthBox = new QSpinBox;
	widthBox->setValue(GLWidget::Width());
	widthBox->setMaximum(1920);
	widthBox->setKeyboardTracking(false);
	//SIGNAL/SLOT if there's parameters
	connect(widthBox, SIGNAL(valueChanged(int)), this, SLOT(WidthChanged(int)));

	heightBox = new QSpinBox;
	heightBox->setValue(GLWidget::Height());
	heightBox->setMaximum(1024);
	heightBox->setKeyboardTracking(false);
	connect(heightBox, SIGNAL(valueChanged(int)), this, SLOT(HeightChanged(int)));

	projMatrixBox = new QComboBox;
	projMatrixBox->addItem("Perspective");
	projMatrixBox->addItem("Orthographic");

	fovBox = new QDoubleSpinBox;
	fovBox->setKeyboardTracking(false);

	aspectBox = new QDoubleSpinBox;
	aspectBox->setKeyboardTracking(false);

	nearBox = new QDoubleSpinBox;
	nearBox->setMaximum(0.0);
	nearBox->setKeyboardTracking(false);

	farBox = new QDoubleSpinBox;
	farBox->setMinimum(0.0);
	farBox->setKeyboardTracking(false);

	leftBox = new QDoubleSpinBox;
	leftBox->setKeyboardTracking(false);

	rightBox = new QDoubleSpinBox;
	rightBox->setKeyboardTracking(false);

	bottomBox = new QDoubleSpinBox;
	bottomBox->setKeyboardTracking(false);

	topBox = new QDoubleSpinBox;
	topBox->setKeyboardTracking(false);

	drawBox = new QComboBox;
	drawBox->addItem("GL_TRIANGLES");
	drawBox->addItem("GL_TRIANGLE_STRIP");
	drawBox->addItem("GL_POINTS");
	connect(drawBox, SIGNAL(currentIndexChanged(int)), this, SLOT(DrawModeChanged(int)));

	colourBox = new ColourTextBox("#6495ED");

	vsyncBox = new QCheckBox;
	vsyncBox->setChecked(true);
	connect(vsyncBox, SIGNAL(clicked(bool)), this, SLOT(VSyncChanged(bool)));

	msaaBox = new QCheckBox;
	connect(msaaBox, SIGNAL(clicked(bool)), this, SLOT(MSAAChanged(bool)));

	mainLayout->addWidget(widthLabel, 0, 0);
	mainLayout->addWidget(widthBox, 0, 1, 1, 7);
	mainLayout->addWidget(heightLabel, 1, 0);
	mainLayout->addWidget(heightBox, 1, 1, 1, 7);
	mainLayout->addWidget(projMatrixLabel, 2, 0);
	mainLayout->addWidget(projMatrixBox, 2, 1, 1, 7);
	mainLayout->addWidget(fovLabel, 3, 0);
	mainLayout->addWidget(fovBox, 3, 1);
	mainLayout->addWidget(aspectLabel, 3, 2);
	mainLayout->addWidget(aspectBox, 3, 3);
	mainLayout->addWidget(nearLabel, 3, 4);
	mainLayout->addWidget(nearBox, 3, 5);
	mainLayout->addWidget(farLabel, 3, 6);
	mainLayout->addWidget(farBox, 3, 7);
	mainLayout->addWidget(drawLabel, 4, 0);
	mainLayout->addWidget(drawBox, 4, 1, 1, 7);
	mainLayout->addWidget(colourLabel, 5, 0);
	mainLayout->addWidget(colourBox, 5, 1, 1, 7);
	mainLayout->addWidget(vsyncLabel, 6, 0);
	mainLayout->addWidget(vsyncBox, 6, 1, 1, 7);
	mainLayout->addWidget(msaaLabel, 7, 0);
	mainLayout->addWidget(msaaBox, 7, 1, 1, 7);

	setLayout(mainLayout);
}

GLGeneralTab::~GLGeneralTab()
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

void GLGeneralTab::WidthChanged(int i)
{
	QMessageBox::information(this, "Changed", "Width Changed");
	GLWidget::Resize(i, GLWidget::Height());
}

void GLGeneralTab::HeightChanged(int i)
{
	QMessageBox::information(this, "Changed", "Height Changed");
	GLWidget::Resize(GLWidget::Width(), i);
}

void GLGeneralTab::ProjMatrixChanged(int i)
{

}

void GLGeneralTab::DrawModeChanged(int i)
{
	//QMessageBox::information(this, "Changed", "Draw Mode Changed");
	GLWidget::DrawMode(i);
}

void GLGeneralTab::VSyncChanged(bool b)
{
	QMessageBox::information(this, "Changed", "VSync Changed");
	GLWidget::VSync(b);
}

void GLGeneralTab::MSAAChanged(bool b)
{
	QMessageBox::information(this, "Changed", "MSAA Changed");
	GLWidget::MSAA(b);
}

