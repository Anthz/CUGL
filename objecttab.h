#ifndef OBJECTTAB_H
#define OBJECTTAB_H

#include <QWidget>
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QPushButton>
#include <QAbstractItemView>
#include <QMessageBox>
#include <QListView>
#include <QStringListModel>
#include <QScrollArea>

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

	QGridLayout* mainLayout;
	QHBoxLayout* buttonLayout;
	QPushButton* add;
	QPushButton* remove;
	QTableWidget* table;
	QStringList objectStringList;
	QStringListModel *listModel;
	QListView *listView;
	QScrollArea *detailScroll;

signals:

public slots:

private slots:
	void ObjectSelected(const QItemSelection& selection);
	void ListEditEnd(QWidget *editor, QAbstractItemDelegate::EndEditHint);
	void TableDoubleClicked();
	void RemoveObject();
	void Popup();
};

#endif // OBJECTTAB_H


