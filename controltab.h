#ifndef CONTROLSTAB_H
#define CONTROLSTAB_H

#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>

class ControlTab : public QWidget
{
	Q_OBJECT

public:
	explicit ControlTab(QWidget* parent = 0);
	~ControlTab();

private:
	QHBoxLayout* mainLayout;
	QPushButton *first, *back, *stop, *play, *forward, *last;

signals:

public slots :

private slots :
	void PlayClicked();
	void StepForward();
	void StepBackward();
	
};

#endif // CONTROLSTAB_H


