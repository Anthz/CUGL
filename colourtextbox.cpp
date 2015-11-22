#include "colourtextbox.h"

ColourTextBox::ColourTextBox(const QString& defaultColour) : QLineEdit(defaultColour)
{
	glFuncs = 0;
	connect(this, &QLineEdit::textChanged, this, &ColourTextBox::TextChanged);
}

ColourTextBox::~ColourTextBox()
{
	
}

void ColourTextBox::mouseDoubleClickEvent(QMouseEvent*)
{
	//open colour picker and set box text to hex
	QColor c = QColorDialog::getColor(text());
	if(c.isValid())
		setText(c.name());
}

void ColourTextBox::TextChanged()
{
	GLWidget::MakeCurrent();

	if(!glFuncs)
		glFuncs = QOpenGLContext::currentContext()->functions();

	//QOpenGLContext::currentContext()->makeCurrent();
	int r, g, b, a;
	QColor c(text());
	c.getRgb(&r, &g, &b, &a);
	glFuncs->glClearColor(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);

	GLWidget::DoneCurrent();
}

