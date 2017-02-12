#include "texturepopup.h"
#include "glsettings.h"

TexturePopup::TexturePopup(QWidget* parent) : QDialog(parent, Qt::WindowTitleHint | Qt::WindowCloseButtonHint)
{
	append = false;
	//Texture detail popup
	//setup layouts/widgets
	mainLayout = new QGridLayout;

	nameLabel = new QLabel("Name/ID:");
	targetLabel = new QLabel("Target:");
	dataLabel = new QLabel("Data:");
	widthLabel = new QLabel("Width:");
	heightLabel = new QLabel("Height:");
	depthLabel = new QLabel("Depth:");
	formatLabel = new QLabel("Format:");
	minMagLabel = new QLabel("Min/Mag Filter:");
	wrapLabel = new QLabel("Wrap Mode:");
	fboLabel = new QLabel("FBO:");

	nameBox = new QLineEdit;
	//SIGNAL/SLOT if there's parameters

	targetBox = new QComboBox;
	targetBox->addItem("GL_TEXTURE_1D");
	targetBox->addItem("GL_TEXTURE_2D");
	targetBox->addItem("GL_TEXTURE_3D");
	targetBox->addItem("GL_TEXTURE_RECTANGLE");
	targetBox->addItem("GL_TEXTURE_CUBE_MAP");
	targetBox->setCurrentIndex(1);
	targetBox->connect(targetBox, SIGNAL(currentIndexChanged(int)), this, SLOT(TargetChanged(int)));

	dataBox = new QLineEdit;
	dataBox->installEventFilter(this);

	widthBox = new QSpinBox;
	widthBox->setMinimum(0);
	widthBox->setMaximum(9999);
	widthBox->setKeyboardTracking(false);

	heightBox = new QSpinBox;
	heightBox->setMinimum(0);
	heightBox->setMaximum(9999);
	heightBox->setKeyboardTracking(false);

	depthBox = new QSpinBox;
	depthBox->setMinimum(0);
	depthBox->setMaximum(9999);
	depthBox->setKeyboardTracking(false);

	formatBox = new QComboBox;
	formatBox->addItem("RGBA");
	formatBox->addItem("RGB");
	formatBox->addItem("8-bit");

	minMagBox = new QComboBox;
	minMagBox->addItem("GL_NEAREST");
	minMagBox->addItem("GL_LINEAR");
	minMagBox->addItem("GL_NEAREST_MIPMAP_NEAREST");
	minMagBox->addItem("GL_LINEAR_MIPMAP_NEAREST");
	minMagBox->addItem("GL_NEAREST_MIPMAP_LINEAR");
	minMagBox->addItem("GL_LINEAR_MIPMAP_LINEAR");

	wrapBox = new QComboBox;
	wrapBox->addItem("GL_REPEAT");
	wrapBox->addItem("GL_MIRRORED_REPEAT");
	wrapBox->addItem("GL_CLAMP_TO_EDGE");
	wrapBox->addItem("GL_CLAMP_TO_BORDER");
	wrapBox->addItem("GL_MIRROR_CLAMP_TO_EDGE (4.4+)");
	wrapBox->setCurrentIndex(2);

	fboBox = new QCheckBox;

	buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
	connect(buttons, SIGNAL(accepted()), this, SLOT(Save()));
	connect(buttons, SIGNAL(rejected()), this, SLOT(close()));

	mainLayout->addWidget(nameLabel, 0, 0);
	mainLayout->addWidget(nameBox, 0, 1);
	mainLayout->addWidget(targetLabel, 1, 0);
	mainLayout->addWidget(targetBox, 1, 1);
	mainLayout->addWidget(dataLabel, 2, 0);
	mainLayout->addWidget(dataBox, 2, 1);
	mainLayout->addWidget(widthLabel, 3, 0);
	mainLayout->addWidget(widthBox, 3, 1);
	mainLayout->addWidget(heightLabel, 4, 0);
	mainLayout->addWidget(heightBox, 4, 1);
	mainLayout->addWidget(depthLabel, 5, 0);
	mainLayout->addWidget(depthBox, 5, 1);
	mainLayout->addWidget(formatLabel, 6, 0);
	mainLayout->addWidget(formatBox, 6, 1);
	mainLayout->addWidget(minMagLabel, 7, 0);
	mainLayout->addWidget(minMagBox, 7, 1);
	mainLayout->addWidget(wrapLabel, 8, 0);
	mainLayout->addWidget(wrapBox, 8, 1);
	mainLayout->addWidget(fboLabel, 9, 0);
	mainLayout->addWidget(fboBox, 9, 1);
	mainLayout->addWidget(buttons, 10, 1);

	setLayout(mainLayout);

	TargetChanged(targetBox->currentIndex());
}

TexturePopup::TexturePopup(QWidget* parent, Texture *t) : TexturePopup(parent)
{
	append = true;
	appBuf = t;

	nameBox->setText(t->Name());
	targetBox->setCurrentIndex(targetBox->findText(t->Target().second));
	dataBox->setText(t->DataPath());
	widthBox->setValue(t->ImageSize().width());
	heightBox->setValue(t->ImageSize().height());
	//depthBox

	UpdateFormatBox(t->Image().format());

	minMagBox->setCurrentIndex(minMagBox->findText(t->MinMagFilter().second));
	wrapBox->setCurrentIndex(wrapBox->findText(t->WrapMode().second));
	fboBox->setChecked(t->FBO());
}

TexturePopup::~TexturePopup()
{
 	delete mainLayout;
}

bool TexturePopup::Validation()
{
	bool result = true;

	if(nameBox->text().isEmpty())
	{
		nameBox->setStyleSheet("border: 2px solid red");
		result = false;
	}
	else
	{
		nameBox->setStyleSheet("");
	}

	/*if(!dataBox->text().isEmpty())
	{
		nameBox->setStyleSheet("border: 2px solid red");
		result = false;
	}
	else
	{
		nameBox->setStyleSheet("");
	}*/

	return result;
}

bool TexturePopup::eventFilter(QObject* object, QEvent* event)
{
	if(object == dataBox && event->type() == QEvent::MouseButtonDblClick)
	{
		CustomDataClicked();
		return true; //stops event continuing
	}
	return false;
}

//switch gui mode
void TexturePopup::TargetChanged(int i)
{
	switch(i)
	{
	case 0:
		heightLabel->hide();
		heightBox->hide();
		depthLabel->hide();
		depthBox->hide();
		break;
	case 1:
		heightLabel->show();
		heightBox->show();
		depthLabel->hide();
		depthBox->hide();
		break;
	case 2:
		heightLabel->show();
		heightBox->show();
		depthLabel->show();
		depthBox->show();
		break;
	case 3:
		break;
	case 4:
		break;
	}
}

void TexturePopup::UpdateFormatBox(QImage::Format fmt)
{
	switch(fmt)
	{
	case QImage::Format_ARGB32:
		formatBox->setCurrentIndex(0);
		break;
	case QImage::Format_RGB888:
		formatBox->setCurrentIndex(1);
		break;
	case QImage::Format_Grayscale8:
		formatBox->setCurrentIndex(2);
		break;
	}
}

QImage::Format TexturePopup::GetFormat()
{
	QImage::Format fmt;

	switch(formatBox->currentIndex())
	{
	case 0:
		fmt = QImage::Format_ARGB32;
		break;
	case 1:
		fmt = QImage::Format_RGB888;
		break;
	case 2:
		fmt = QImage::Format_Grayscale8;
		break;
	}

	return fmt;
}

void TexturePopup::CustomDataClicked()
{
	QString title = "Open Image File";
	QString filter = "Image Files (*.png *.jpg *.bmp)";

	path = QFileDialog::getOpenFileName(this, title, QDir::currentPath(), filter);
	if(path.size() == 0)
		return; //cancelled

	dataBox->setText(path);
	dataBox->setCursorPosition(path.size());

	int nameBegin = path.lastIndexOf("/") + 1;
	//int nameEnd = s.lastIndexOf(".") + 1; //to remove extension

	QString name = path.right(path.length() - nameBegin);	//only used for error message

	QImage img = QImage(path);	//temp image instead of passing it through (makes saving easier)
	if(!img.isNull())
	{
		widthBox->setValue(img.size().width());
		heightBox->setValue(img.size().height());
		UpdateFormatBox(img.format());
	}
	else
	{
		Logger::Log("Failed to load texture " + name.toStdString());
	}
}

void TexturePopup::Save()
{
	if(Validation())
	{
		QString name = nameBox->text();
		QString dataPath = dataBox->text();

		//default to full res
		int width = widthBox->value();
		if(width == 0)
			width = GLWidget::Width();

		int height = heightBox->value();
		if(height == 0)
			height = GLWidget::Height();

		target = targetBox->currentText();
		minMagFilter = minMagBox->currentText();
		wrapMode = wrapBox->currentText();
		QImage::Format fmt = GetFormat();

		bool fbo = fboBox->isChecked();

		if(!append)
		{
			Texture *t;

			t = new Texture(name, dataPath, width, height, fmt, target, minMagFilter, wrapMode, fbo);
			GLSettings::TextureList.push_back(t);
			static_cast<TextureTab*>(parent())->AddToList(t);
		}
		else
		{
			/*appBuf->Name(name);
			appBuf->Target(target);
			appBuf->Image = appBuf->Image.convertToFormat(fmt);
			if(dataPath != "")
			{
				appBuf->DataPath(dataPath);
				appBuf->Image(img);
			}
			appBuf->ImageSize(QSize(width, height));
			appBuf->MinMagFilter(minMagFilter);
			appBuf->WrapMode(wrapMode);
			appBuf->FBO(fbo);*/
		}

		close();
	}
}