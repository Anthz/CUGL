#ifndef OBJECTTAB_H
#define OBJECTTAB_H

#include <QWidget>
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QPushButton>
#include <QAbstractItemView>
#include <QMessageBox>

#include "object.h"
#include "cuglbuffer.h"
#include "objectpopup.h"

class ObjectTab : public QWidget
{
	Q_OBJECT

public:
	explicit ObjectTab(QWidget* parent = 0);
	~ObjectTab();

	void AddToTable(Object* o);

private:
	QString BuffersToString(std::vector<CUGLBuffer*> *buffers);

	QVBoxLayout* mainLayout;
	QHBoxLayout* buttonLayout;
	QPushButton* add;
	QPushButton* remove;
	QTableWidget* table;

signals:

public slots:

private slots:
	void TableDoubleClicked();
	void RemoveObject();
	void Popup();
};

#endif // OBJECTTAB_H


