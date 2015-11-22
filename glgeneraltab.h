#ifndef GLGENERALTAB_H
#define GLGENERALTAB_H

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

class GLGeneralTab : public QWidget
{
	Q_OBJECT

public:
	explicit GLGeneralTab(QWidget* parent = 0);
	~GLGeneralTab();

private:
	QGridLayout* mainLayout;
	QLabel *widthLabel, *heightLabel,
		   *projMatrixLabel, *drawLabel,
		   *colourLabel, *vsyncLabel,
		   *msaaLabel, *fovLabel;
	QSpinBox *widthBox, *heightBox,
	         *fovBox;
	QComboBox *projMatrixBox, *drawBox;
	ColourTextBox *colourBox;
	QCheckBox *vsyncBox, *msaaBox;
signals:

public slots:
	void WidthChanged(int i);
	void HeightChanged(int i);
	void FOVChanged(int i);
	void ProjMatrixChanged(int i);
	void DrawModeChanged(int i);
	void VSyncChanged(bool b);
	void MSAAChanged(bool b);

private slots:
};

#endif // GLGENERALTAB_H


