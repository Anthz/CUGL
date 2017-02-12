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

struct Device
{
	char name[100];
	int handle;
};

class CUGeneralTab : public QWidget
{
	Q_OBJECT

public:
	explicit CUGeneralTab(QWidget* parent = 0);
	~CUGeneralTab();

	void UpdateDevices(std::vector<Device>& dList);

private:
	QGridLayout* mainLayout;
	QLabel *deviceLabel;
	QComboBox* deviceBox;

signals:

public slots:
private slots:
	void DeviceChanged(int i);
};

#endif // CUGENERALTAB_H


