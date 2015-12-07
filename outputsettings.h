#ifndef OUTPUTSETTINGS_H
#define OUTPUTSETTINGS_H

#include <QWidget>
#include <QVBoxLayout>
#include <QGroupBox>

class OutputSettings : public QWidget
{
	Q_OBJECT
	
public:
	explicit OutputSettings(QWidget* parent = 0);
	~OutputSettings();

	virtual QSize minimumSizeHint() const override;
	virtual QSize sizeHint() const override;

private:
	QVBoxLayout* mainLayout;
	QGridLayout* settingsLayout;
	QGroupBox* settingsGroup;

signals:

public slots:
};

#endif // OUTPUTSETTINGS_H


