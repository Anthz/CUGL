#ifndef OBJECTPOPUP_H
#define OBJECTPOPUP_H

#include <QWidget>
#include <QDialog>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QDialogButtonBox>
#include <QStandardItemModel>
#include <QStandardItem>
#include <QListView>
#include <QEvent>
#include <iostream>

#include "cuglbuffer.h"
#include "object.h"

class ObjectPopup : public QDialog
{
	Q_OBJECT

public:
	ObjectPopup(QWidget* parent);
	ObjectPopup(QWidget* parent, Object* o);
	~ObjectPopup();

private:
	bool Validation();

	QGridLayout* mainLayout;
	QLabel *nameLabel, *instancesLabel,
		*bufferLabel, *textureLabel, *shaderLabel;
	QLineEdit* nameBox;
	QSpinBox* instancesBox;
	QComboBox *bufferBox, *textureBox, *shaderBox;

	QDialogButtonBox* buttons;
	QStandardItemModel* bufferBoxModel;
	std::vector<QStandardItem*> itemList;
	std::vector<CUGLBuffer*> buffers;

	bool append;
	Object *appObj;

private slots:
	void BuffersChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight);
	void Save();
};

#endif // OBJECTPOPUP_H


