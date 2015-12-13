#include "bufferpopup.h"
#include "glsettings.h"

#pragma region Shapes
float quad[] = 
{
	-0.5f, -0.5f, 0.0f,
	-0.5f, 0.5f, 0.0f,
	0.5f, 0.5f, 0.0f,
	0.5f, 0.5f, 0.0f,
	0.5f, -0.5f, 0.0f,
	-0.5f, -0.5f, 0.0f
};

float quadUV[] =
{
	0.0f, 1.0f,
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f
};

GLfloat triangle[] =
{
	-1.0f, -1.0f, 0.0f,
	0.0f, 1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
};
#pragma endregion Shapes

BufferPopup::BufferPopup(QWidget* parent) : QDialog(parent, Qt::WindowTitleHint | Qt::WindowCloseButtonHint)
{
	append = false;
	//Object detail popup
	//setup layouts/widgets
	mainLayout = new QGridLayout;

	nameLabel = new QLabel("Name/ID:");
	targetLabel = new QLabel("Target:");
	capacityLabel = new QLabel("Capacity:");
	dataLabel = new QLabel("Data:");
	usageLabel = new QLabel("Usage:");
	attribNameLabel = new QLabel("Attribute Name:");
	attribCapacityLabel = new QLabel("Attribute Capacity:");
	typeLabel = new QLabel("Type:");
	normalisedLabel = new QLabel("Normalised:");

	nameBox = new QLineEdit;
	//SIGNAL/SLOT if there's parameters

	targetBox = new QComboBox;
	targetBox->addItem("GL_ARRAY_BUFFER");
	targetBox->addItem("GL_PIXEL_UNPACK_BUFFER");
	targetBox->connect(targetBox, SIGNAL(currentIndexChanged(int)), this, SLOT(TargetChanged(int)));

	capacityBox = new QSpinBox;
	capacityBox->setMinimum(1);
	capacityBox->setMaximum(INT_MAX);
	capacityBox->setKeyboardTracking(false);

	dataPickerBox = new QComboBox;
	dataPickerBox->addItem("Custom");
	dataPickerBox->addItem("Screen Aligned Quad Vertices");
	dataPickerBox->addItem("Screen Aligned Quad UV");
	dataPickerBox->connect(dataPickerBox, SIGNAL(currentIndexChanged(int)), this, SLOT(DataChanged(int)));

	dataBox = new QLineEdit;
	dataBox->installEventFilter(this);

	textureBox = new QComboBox;
	for(int i = 0; i < GLSettings::TextureList.size(); ++i)
	{
		textureBox->addItem(GLSettings::TextureList.at(i)->Name());
	}
	textureBox->hide();

	usageBox = new QComboBox;
	usageBox->addItem("GL_DYNAMIC_DRAW");
	usageBox->addItem("GL_DYNAMIC_COPY");
	usageBox->addItem("GL_STATIC_DRAW");
	usageBox->addItem("GL_STATIC_COPY");

	attribNameBox = new QLineEdit;

	attribCapacityBox = new QSpinBox;
	attribCapacityBox->setMinimum(1);
	attribCapacityBox->setMaximum(999999);
	attribCapacityBox->setKeyboardTracking(false);

	typeBox = new QComboBox;
	typeBox->addItem("GL_FLOAT");
	typeBox->addItem("GL_HALF_FLOAT");
	typeBox->addItem("GL_DOUBLE");
	typeBox->addItem("GL_INT");
	typeBox->addItem("GL_UNSIGNED_INT");
	typeBox->addItem("GL_SHORT");
	typeBox->addItem("GL_UNSIGNED_SHORT");
	typeBox->addItem("GL_BYTE");
	typeBox->addItem("GL_UNSIGNED_BYTE");

	normalisedBox = new QCheckBox;

	buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
	connect(buttons, SIGNAL(accepted()), this, SLOT(Save()));
	connect(buttons, SIGNAL(rejected()), this, SLOT(close()));

	mainLayout->addWidget(nameLabel, 0, 0);
	mainLayout->addWidget(nameBox, 0, 1);
	mainLayout->addWidget(targetLabel, 1, 0);
	mainLayout->addWidget(targetBox, 1, 1);
	mainLayout->addWidget(capacityLabel, 2, 0);
	mainLayout->addWidget(capacityBox, 2, 1);
	mainLayout->addWidget(dataLabel, 3, 0);
	mainLayout->addWidget(dataPickerBox, 3, 1);
	mainLayout->addWidget(textureBox, 3, 1);
	mainLayout->addWidget(dataBox, 4, 1);
	mainLayout->addWidget(usageLabel, 5, 0);
	mainLayout->addWidget(usageBox, 5, 1);
	mainLayout->addWidget(attribNameLabel, 6, 0);
	mainLayout->addWidget(attribNameBox, 6, 1);
	mainLayout->addWidget(attribCapacityLabel, 7, 0);
	mainLayout->addWidget(attribCapacityBox, 7, 1);
	mainLayout->addWidget(typeLabel, 8, 0);
	mainLayout->addWidget(typeBox, 8, 1);
	mainLayout->addWidget(normalisedLabel, 9, 0);
	mainLayout->addWidget(normalisedBox, 9, 1);
	mainLayout->addWidget(buttons, 10, 1);

	setLayout(mainLayout);
}

BufferPopup::BufferPopup(QWidget* parent, CUGLBuffer *b) : BufferPopup(parent)
{
	append = true;
	appBuf = b;

	nameBox->setText(b->bName);
	capacityBox->setValue(b->bCap);
	targetBox->setCurrentIndex(targetBox->findText(b->bTarget.second));

	if(b->bName == "Screen Aligned Quad Vertices")
	{
		dataPickerBox->setCurrentIndex(1);
		DisableBufferBoxes(true);
	}
	else if(b->bName == "Screen Aligned Quad UVs")
	{
		dataPickerBox->setCurrentIndex(2);
		DisableBufferBoxes(true);
	}
	else
	{
		dataPickerBox->setCurrentIndex(0);
		dataBox->setText(b->bDataPath);
		DisableBufferBoxes(false);
	}
	
	usageBox->setCurrentIndex(usageBox->findText(b->bUsage.second));
	attribNameBox->setText(b->aID.first);
	attribCapacityBox->setValue(b->aSize);
	typeBox->setCurrentIndex(typeBox->findText(std::get<1>(b->bType)));
	normalisedBox->setChecked(b->norm);
}

BufferPopup::~BufferPopup()
{
	delete nameLabel;
	delete targetLabel;
	delete capacityLabel;
	delete dataLabel;
	delete usageLabel;
	delete attribNameLabel;
	delete attribCapacityLabel;
	delete typeLabel;
	delete normalisedLabel;

	delete nameBox;
	delete dataPickerBox;
	delete dataBox;
	delete textureBox;
	delete attribNameBox;
	delete capacityBox;
	delete attribCapacityBox;
	delete targetBox;
	delete usageBox;
	delete typeBox;
	delete normalisedBox;

	delete buttons;

	delete mainLayout;
}

bool BufferPopup::Validation()
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
	
	//set bool for correct custom load
	//if not empty, check for successful load
	//if blank, set data as 0
	/*if(dataBox->text().isEmpty())	//or if file fails to load
	{
		dataBox->setStyleSheet("border: 2px solid red");
		result = false;
	}
	else
	{
		dataBox->setStyleSheet("");
	}*/

	if(attribNameBox->text().isEmpty())	//if string, check returned id
	{
		attribNameBox->setStyleSheet("border: 2px solid red");
		result = false;
	}
	else
	{
		attribNameBox->setStyleSheet("");
	}

	return result;
}

bool BufferPopup::eventFilter(QObject* object, QEvent* event)
{
	if(object == dataBox && event->type() == QEvent::MouseButtonDblClick)
	{
		CustomDataClicked();
		return true; //stops event continuing
	}
	return false;
}

void BufferPopup::DisableBufferBoxes(bool b)
{
	nameBox->setEnabled(!b);
	capacityBox->setEnabled(!b);
	dataBox->setHidden(b);
	targetBox->setEnabled(!b);
	attribCapacityBox->setEnabled(!b);
	typeBox->setEnabled(!b);
}

//switch gui mode
void BufferPopup::TargetChanged(int i)
{
	switch(i)
	{
		case 0:
		{
			dataLabel->setText("Data:");
			dataPickerBox->show();
			dataBox->show();

			capacityLabel->show();
			capacityBox->show();

			textureBox->hide();
			break;
		}
		case 1:
		{
			dataLabel->setText("Texture:");
			dataPickerBox->hide();
			dataBox->hide();

			capacityLabel->hide();
			capacityBox->hide();

			textureBox->show();
			break;
		}
	}
}

void BufferPopup::SetTarget()
{
	switch(targetBox->currentIndex())
	{
		case 0:
		{
			target.first = GL_ARRAY_BUFFER;
			break;
		}
		case 1:
		{
			target.first = GL_PIXEL_UNPACK_BUFFER;
			break;
		}
	}

	target.second = targetBox->currentText();
}

void BufferPopup::SetData()
{
	if(targetBox->currentIndex() == 0)
	{
		switch(dataPickerBox->currentIndex())
		{
			case 1:
			{
				data = quad;
				break;
			}
			case 2:
			{
				data = quadUV;
				break;
			}
		}
	}
	else
	{
		Texture *t = GLSettings::TextureList.at(textureBox->currentIndex());
		capacityBox->setValue(t->ImageSize().width() * t->ImageSize().height() * 4);
		data = nullptr;//(void*)t->Data();
	}
}

void BufferPopup::DataChanged(int i)
{
	switch(i)
	{
		case 0:
			nameBox->setText("");
			capacityBox->setValue(1);
			attribNameBox->setText("");
			attribCapacityBox->setValue(0);

			DisableBufferBoxes(false);
			CustomDataClicked();
			break;
		case 1:
			//SAQ Verts
			nameBox->setText("Screen Aligned Quad Vertices");
			capacityBox->setValue(18);
			targetBox->setCurrentIndex(0);
			dataBox->setText("");
			usageBox->setCurrentIndex(0);
			attribNameBox->setText("aPos");
			attribCapacityBox->setValue(3);
			typeBox->setCurrentIndex(0);
			normalisedBox->setChecked(false);

			DisableBufferBoxes(true);
			break;
		case 2:
			//SAQ UVs
			nameBox->setText("Screen Aligned Quad UVs");
			capacityBox->setValue(12);
			targetBox->setCurrentIndex(0);
			dataBox->setText("");
			usageBox->setCurrentIndex(0);
			attribNameBox->setText("aUV");
			attribCapacityBox->setValue(2);
			typeBox->setCurrentIndex(0);
			normalisedBox->setChecked(false);

			DisableBufferBoxes(true);
			break;
	}
}

void BufferPopup::CustomDataClicked()
{
	QString s;
	QString title = "Open Data File";
	QString filter = "";
		
	s = QFileDialog::getOpenFileName(this, title, QDir::currentPath(), filter);
	if(s.size() == 0)
		return; //cancelled

	dataBox->setText(s);
	dataBox->setCursorPosition(s.size());
}

void BufferPopup::SetUsage()
{
	switch(usageBox->currentIndex())
	{
		case 0:
			usage.first = GL_DYNAMIC_DRAW;
			break;
		case 1:
			usage.first = GL_DYNAMIC_COPY;
			break;
		case 2:
			usage.first = GL_STATIC_DRAW;
			break;
		case 3:
			usage.first = GL_STATIC_COPY;
			break;
	}

	usage.second = usageBox->currentText();
}

void BufferPopup::SetType()
{
	GLenum typeEnum;
	QString typeString = typeBox->currentText();
	int size;

	switch(typeBox->currentIndex())
	{
		case 0:
			typeEnum = GL_FLOAT;
			size = sizeof(float);
			break;
		case 1:
			typeEnum = GL_HALF_FLOAT;
			size = sizeof(float) / 2;
			break;
		case 2:
			typeEnum = GL_DOUBLE;
			size = sizeof(double);
			break;
		case 3:
			typeEnum = GL_INT;
			size = sizeof(int);
			break;
		case 4:
			typeEnum = GL_UNSIGNED_INT;
			size = sizeof(unsigned int);
			break;
		case 5:
			typeEnum = GL_SHORT;
			size = sizeof(short);
			break;
		case 6:
			typeEnum = GL_UNSIGNED_SHORT;
			size = sizeof(unsigned short);
			break;
		case 7:
			typeEnum = GL_BYTE;
			size = sizeof(byte);
			break;
		case 8:
			typeEnum = GL_UNSIGNED_BYTE;
			size = sizeof(byte);
			break;
	}

	type = std::make_tuple(typeEnum, typeString, size);
}

void BufferPopup::Save()
{
	if(Validation())
	{
		SetTarget();
		SetData();
		SetUsage();
		SetType();

		QString name = nameBox->text();
		int capacity = capacityBox->value();
		QString dataPath = dataBox->text();
		std::pair<QString, int> aID(attribNameBox->text(), -1);
		int aCapacity = attribCapacityBox->value();
		bool norm = normalisedBox->isChecked();

		if(!append)
		{
			CUGLBuffer* b;

			if(dataPickerBox->isVisible() && dataPickerBox->currentIndex() == 0)
				b = new CUGLBuffer(name, capacity, target, dataPath, usage, aID, aCapacity, type, norm);
			else
				b = new CUGLBuffer(name, capacity, target, data, usage, aID, aCapacity, type, norm);
			if(targetBox->currentIndex() == 1)
				GLSettings::TextureList.at(textureBox->currentIndex())->PBO(b->bufID);
			GLSettings::BufferList.push_back(b);
			static_cast<GLBufferTab*>(parent())->AddToTable(b);
		}
		else
		{
			appBuf->bName = name;
			appBuf->bCap = capacity;
			appBuf->bTarget = target;
			if(dataPath != "")
				appBuf->bDataPath = dataPath;
			else
				appBuf->bData = data;
			appBuf->bUsage = usage;
			appBuf->aID = aID;
			appBuf->aSize = aCapacity;
			appBuf->bType = type;
			appBuf->norm = norm;
		}

		close();
	}
}

