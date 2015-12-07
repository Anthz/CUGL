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
	void SetTarget();
	void SetMinMagFilter();
	void SetWrapMode();

	QWidget* parentWidget;
	QGridLayout* mainLayout;
	QLabel *nameLabel, *targetLabel,
		*dataLabel, *widthLabel,
		*heightLabel, *depthLabel,
		*minMagLabel, *wrapLabel,
		*fboLabel;

	QLineEdit *nameBox, *dataBox;
	QSpinBox *widthBox, *heightBox, *depthBox;
	QComboBox *targetBox, *minMagBox, *wrapBox;
	QCheckBox *fboBox;
	QDialogButtonBox* buttons;

	QImage img;
	std::pair<GLenum, QString> target;
	std::pair<GLenum, QString> minMagFilter, wrapMode;

	bool append;
	Texture *appBuf;

private slots:
	bool eventFilter(QObject* object, QEvent* event);
	void TargetChanged(int i);
	void Save();
};

#endif // TEXTUREPOPUP_H


