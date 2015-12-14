#ifndef GLBUFFERTAB_H
#define GLBUFFERTAB_H

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

#include "cuglbuffer.h"
#include "bufferpopup.h"

class GLBufferTab : public QWidget
{
	Q_OBJECT

public:
	explicit GLBufferTab(QWidget* parent = 0);
	~GLBufferTab();

	void AddToTable(CUGLBuffer* b);

private:
	QGridLayout *mainLayout;
	QHBoxLayout *bufferLayout, *buttonLayout;
	QPushButton* add;
	QPushButton* remove;
	QTableWidget* table;
	QStringList bufferStringList;
	QStringListModel *listModel;
	QListView *listView;
	QScrollArea *detailScroll;
signals:

public slots:

private slots:
	void BufferSelected(const QItemSelection& selection);
	void ListEditEnd(QWidget *editor, QAbstractItemDelegate::EndEditHint);
	void TableDoubleClicked();
	void RemoveBuffer();
	void Popup();
};

#endif // GLBUFFERTAB_H


