#ifndef GLBUFFERTAB_H
#define GLBUFFERTAB_H

#include <QWidget>
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QPushButton>
#include <QAbstractItemView>
#include <QMessageBox>

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
	QVBoxLayout* mainLayout;
	QHBoxLayout* buttonLayout;
	QPushButton* add;
	QPushButton* remove;
	QTableWidget* table;

	signals:

public slots:

private slots:
	void TableDoubleClicked();
	void RemoveBuffer();
	void Popup();
};

#endif // GLBUFFERTAB_H


