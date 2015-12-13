#ifndef CUGENERALTAB_H
#define CUGENERALTAB_H

#include <QWidget>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QComboBox>
#include <QColor>
#include <QColorDialog>
#include <QCheckBox>
#include <QEvent>

#include "colourtextbox.h"
#include "serializable.h"

class CUGeneralTab : public QWidget, public Serializable
{
	Q_OBJECT

public:
	explicit CUGeneralTab(QWidget* parent = 0);
	~CUGeneralTab();

private:
	QGridLayout* mainLayout;
	QLabel *widthLabel, *heightLabel,
	       *drawLabel, *colourLabel,
	       *vsyncLabel, *msaaLabel,
	       *fovLabel;
	QSpinBox *widthBox, *heightBox,
	         *fovBox;
	QComboBox* drawBox;
	ColourTextBox* colourBox;
	QCheckBox *vsyncBox, *msaaBox;
signals:

public slots:
private slots:
	void WidthChanged(int i);
};

#endif // CUGENERALTAB_H


