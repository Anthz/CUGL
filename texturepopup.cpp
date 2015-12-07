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
	widthBox->setMinimum(1);
	widthBox->setMaximum(9999);
	widthBox->setKeyboardTracking(false);

	heightBox = new QSpinBox;
	heightBox->setMinimum(1);
	heightBox->setMaximum(9999);
	heightBox->setKeyboardTracking(false);

	depthBox = new QSpinBox;
	depthBox->setMinimum(1);
	depthBox->setMaximum(9999);
	depthBox->setKeyboardTracking(false);

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
	mainLayout->addWidget(minMagLabel, 6, 0);
	mainLayout->addWidget(minMagBox, 6, 1);
	mainLayout->addWidget(wrapLabel, 7, 0);
	mainLayout->addWidget(wrapBox, 7, 1);
	mainLayout->addWidget(fboLabel, 8, 0);
	mainLayout->addWidget(fboBox, 8, 1);
	mainLayout->addWidget(buttons, 9, 1);

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
	minMagBox->setCurrentIndex(minMagBox->findText(t->MinMagFilter().second));
	wrapBox->setCurrentIndex(wrapBox->findText(t->WrapMode().second));
	fboBox->setChecked(t->FBO());
}

TexturePopup::~TexturePopup()
{
	delete nameLabel;
	delete targetLabel;
	delete dataLabel;
	delete widthLabel;
	delete heightLabel;
	delete depthLabel;
	delete minMagLabel;
	delete wrapLabel;
	delete fboLabel;

	delete nameBox;
	delete dataBox;
	delete widthBox;
	delete heightBox;
	delete depthBox;
	delete targetBox;
	delete minMagBox;
	delete wrapBox;
	delete fboBox;

	delete buttons;

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

	if(!dataBox->text().isEmpty() && img.isNull())
	{
		nameBox->setStyleSheet("border: 2px solid red");
		result = false;
	}
	else
	{
		nameBox->setStyleSheet("");
	}

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

void TexturePopup::SetTarget()
{
	switch(targetBox->currentIndex())
	{
		case 0:
		{
			target.first = GL_TEXTURE_1D;
			break;
		}
		case 1:
		{
			target.first = GL_TEXTURE_2D;
			break;
		}
		case 2:
		{
			target.first = GL_TEXTURE_3D;
			break;
		}
		case 3:
		{
			target.first = GL_TEXTURE_RECTANGLE;
			break;
		}
		case 4:
		{
			target.first = GL_TEXTURE_CUBE_MAP;
			break;
		}
	}

	target.second = targetBox->currentText();
}

void TexturePopup::SetMinMagFilter()
{
	switch(minMagBox->currentIndex())
	{
		case 0:
		{
			minMagFilter.first = GL_NEAREST;
			break;
		}
		case 1:
		{
			minMagFilter.first = GL_LINEAR;
			break;
		}
		case 2:
		{
			minMagFilter.first = GL_NEAREST_MIPMAP_NEAREST;
			break;
		}
		case 3:
		{
			minMagFilter.first = GL_LINEAR_MIPMAP_NEAREST;
			break;
		}
		case 4:
		{
			minMagFilter.first = GL_NEAREST_MIPMAP_LINEAR;
			break;
		}
		case 5:
		{
			minMagFilter.first = GL_LINEAR_MIPMAP_LINEAR;
			break;
		}
	}

	minMagFilter.second = minMagBox->currentText();
}

void TexturePopup::SetWrapMode()
{
	switch(wrapBox->currentIndex())
	{
		case 0:
		{
			wrapMode.first = GL_REPEAT;
			break;
		}
		case 1:
		{
			wrapMode.first = GL_MIRRORED_REPEAT;
			break;
		}
		case 2:
		{
			wrapMode.first = GL_CLAMP_TO_EDGE;
			break;
		}
		case 3:
		{
			wrapMode.first = GL_CLAMP_TO_BORDER;
			break;
		}
		case 4:
		{
			wrapMode.first = GL_MIRROR_CLAMP_TO_EDGE;
			break;
		}
	}

	wrapMode.second = wrapBox->currentText();
}

void TexturePopup::CustomDataClicked()
{
	QString s;
	QString title = "Open Image File";
	QString filter = "Image Files (*.png *.jpg *.bmp)";

	s = QFileDialog::getOpenFileName(this, title, QDir::currentPath(), filter);
	if(s.size() == 0)
		return; //cancelled

	dataBox->setText(s);
	dataBox->setCursorPosition(s.size());

	int nameBegin = s.lastIndexOf("/") + 1;
	//int nameEnd = s.lastIndexOf(".") + 1; //to remove extension

	QString name = s.right(s.length() - nameBegin);

	img = QImage(s);
	if(!img.isNull())
	{
		widthBox->setValue(img.size().width());
		heightBox->setValue(img.size().height());
	}
	else
	{
		Logger::Log("Faied to load texture " + name.toStdString());
	}
}

void TexturePopup::Save()
{
	if(Validation())
	{
		SetTarget();
		SetMinMagFilter();
		SetWrapMode();

		QString name = nameBox->text();
		QString dataPath = dataBox->text();
		int width = widthBox->value();
		int height = heightBox->value();
		bool fbo = fboBox->isChecked();

		if(!append)
		{
			Texture *t;

			if(!img.isNull())
				t = new Texture(name, dataPath, img, width, height, target, minMagFilter, wrapMode, fbo);
			else
				t = new Texture(name, width, height, target, minMagFilter, wrapMode, fbo);

			GLSettings::TextureList.push_back(t);
			static_cast<TextureTab*>(parent())->AddToList(t);
		}
		else
		{
			appBuf->Name(name);
			appBuf->Target(target);
			if(dataPath != "")
			{
				appBuf->DataPath(dataPath);
				appBuf->Image(img);
			}
			appBuf->ImageSize(QSize(width, height));
			appBuf->MinMagFilter(minMagFilter);
			appBuf->WrapMode(wrapMode);
			appBuf->FBO(fbo);
		}

		close();
	}
}