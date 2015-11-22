#ifndef COLOURTEXTBOX_H
#define COLOURTEXTBOX_H

#include <QLineEdit>
#include <QMessageBox>
#include <QColor>
#include <QColorDialog>
#include <QOpenGLFunctions>
#include <QSurface>

#include "glwidget.h"

class ColourTextBox : public QLineEdit
{
public:
	ColourTextBox(const QString& defaultColour);
	~ColourTextBox();

private:
	QOpenGLFunctions* glFuncs;

protected:
	void mouseDoubleClickEvent(QMouseEvent*);

private slots:
	void TextChanged();
};

#endif // COLOURTEXTBOX_H


