#ifndef BUFFERPOPUP_H
#define BUFFERPOPUP_H

#include <QWidget>
#include <QDialog>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QEvent>
#include <QFileDialog>
#include <QtGui/qopengl.h>

#include "cuglbuffer.h"

class BufferPopup : public QDialog
{
	Q_OBJECT

public:
	BufferPopup(QWidget* parent);
	BufferPopup(QWidget* parent, CUGLBuffer *b);
	~BufferPopup();

private:
	void ImageGUI(bool b);
	void CustomDataClicked();
	void DisableBufferBoxes(bool b);
	bool Validation();
	void SetTarget();
	void SetData();
	void SetUsage();
	void SetType();

	QWidget* parentWidget;
	QGridLayout* mainLayout;
	QLabel *nameLabel, *targetLabel,
		   *capacityLabel, *dataLabel,
	       *usageLabel, *attribNameLabel,
	       *attribCapacityLabel, *typeLabel,
	       *normalisedLabel;
	//*handledLabel;
	QLineEdit *nameBox, *dataBox, *attribNameBox;
	QSpinBox *capacityBox, *attribCapacityBox;
	QComboBox *targetBox, *dataPickerBox, *usageBox, *typeBox;
	QCheckBox* normalisedBox; //*handledBox;
	QDialogButtonBox* buttons;

	void *data;
	std::pair<GLenum, QString> target, usage;
	std::tuple<GLenum, QString, int> type;

	bool append;
	CUGLBuffer *appBuf;

private slots:
	bool eventFilter(QObject* object, QEvent* event);
	void TargetChanged(int i);
	void DataChanged(int i);
	void Save();
};

#endif // BUFFERPOPUP_H


