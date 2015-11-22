#ifndef PARAMSTAB_H
#define PARAMSTAB_H

#include <QWidget>
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QPushButton>
#include <QMessageBox>

class ParamTab : public QWidget
{
	Q_OBJECT

public:
	explicit ParamTab(QWidget* parent = 0);
	~ParamTab();

private:
	QVBoxLayout* mainLayout;
	QHBoxLayout* buttonLayout;
	QPushButton* add;
	QPushButton* remove;
	QTableWidget *table;

signals:

public slots:
	void AddClicked();
	void RemoveClicked();
};

#endif // PARAMSTAB_H


