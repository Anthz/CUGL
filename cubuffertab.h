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

	void Update();

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

	public slots :

		private slots :
		void BufferChanged(int i);
	void ActivatedChanged(int i);
	void ParamIDChanged(int i);
};


