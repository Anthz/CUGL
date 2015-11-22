#ifndef CUSETTINGS_H
#define CUSETTINGS_H

#include <QWidget>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QTabWidget>

#include "cugeneraltab.h"
#include "glbuffertab.h"
#include "paramtab.h"
#include "cubuffertab.h"
#include "kerneltab.h"

class CUSettings : public QWidget
{
	Q_OBJECT

public:
	struct Device
	{
		char name[100];
		int handle;
	};

	explicit CUSettings(QWidget* parent = 0);
	~CUSettings();

	static std::vector<CUGLBuffer*> BufferList;

	std::vector<Device> deviceList;

private:
	void CUInit();

	QVBoxLayout *mainLayout, *settingsLayout;
	QGroupBox* settingsGroup;
	QTabWidget* tabs;

	int deviceCount;

signals:

	public slots :

		private slots :
		void TabChanged(int i);
};

#endif // CUSETTINGS_H


