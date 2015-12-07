#include "outputsettings.h"

OutputSettings::OutputSettings(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QVBoxLayout();
	settingsGroup = new QGroupBox("Output Settings");

	settingsLayout = new QGridLayout();

	settingsGroup->setLayout(settingsLayout);
	mainLayout->addWidget(settingsGroup);

	setLayout(mainLayout);
}

OutputSettings::~OutputSettings()
{
	delete settingsLayout;
	delete settingsGroup;
	delete mainLayout;
}

QSize OutputSettings::minimumSizeHint() const
{
	return QSize(500, 350);
}

QSize OutputSettings::sizeHint() const
{
	int w = (int)static_cast<QWidget*>(parent())->width() / 3;
	return QSize((int)static_cast<QWidget*>(parent())->width() / 3, 350);
}