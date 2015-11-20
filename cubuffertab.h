#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QComboBox>
#include <QSpinBox>

#include "cuglbuffer.h"

class CUBufferTab : public QWidget
{
	Q_OBJECT

public:
	explicit CUBufferTab(QWidget* parent = 0);
	~CUBufferTab();

private:
	QVBoxLayout* mainLayout;
	QHBoxLayout* bufferLayout, *settingsLayout;
	QLabel* bufferLabel, *activatedLabel,
		*paramIDLabel;
	QComboBox* bufferBox;
	QWidget* settings;
	QCheckBox* activateBox;
	QSpinBox* paramIDBox;

signals:

public slots:

private slots:
};


