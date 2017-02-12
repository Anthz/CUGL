#ifndef TEXTUREPOPUP_H
#define TEXTUREPOPUP_H

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
#include <QImage>
#include <QtGui/qopengl.h>

#include "cuglbuffer.h"
#include "utilities.h"

class TexturePopup : public QDialog
{
	Q_OBJECT
public:
	TexturePopup(QWidget* parent);
	TexturePopup(QWidget* parent, Texture *t);
	~TexturePopup();

private:
	void CustomDataClicked();
	bool Validation();
	void UpdateFormatBox(QImage::Format fmt);
	QImage::Format GetFormat();

	QWidget* parentWidget;
	QGridLayout* mainLayout;
	QLabel *nameLabel, *targetLabel,
		*dataLabel, *widthLabel,
		*heightLabel, *depthLabel,
		*formatLabel, *minMagLabel,
		*wrapLabel, *fboLabel;

	QLineEdit *nameBox, *dataBox;
	QSpinBox *widthBox, *heightBox, *depthBox;
	QComboBox *targetBox, *formatBox, *minMagBox, *wrapBox;
	QCheckBox *fboBox;
	QDialogButtonBox* buttons;
	QString target, minMagFilter, wrapMode, path;

	bool append;
	Texture *appBuf;

	private slots:
	bool eventFilter(QObject* object, QEvent* event);
	void TargetChanged(int i);
	void Save();
};

#endif // TEXTUREPOPUP_H


