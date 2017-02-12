#include "bufferpopup.h"
#include "glsettings.h"

BufferPopup::BufferPopup(QWidget* parent) : QDialog(parent, Qt::WindowTitleHint | Qt::WindowCloseButtonHint)
{
	append = false;
	//Object detail popup
	//setup layouts/widgets
	mainLayout = new QGridLayout;
	checkboxLayout = new QHBoxLayout;

	nameLabel = new QLabel("Name/ID:");
	targetLabel = new QLabel("Target:");
	capacityLabel = new QLabel("Capacity:");
	dataLabel = new QLabel("Data:");
	usageLabel = new QLabel("Usage:");
	attribNameLabel = new QLabel("Attribute Name:");
	attribCapacityLabel = new QLabel("Attribute Capacity:");
	typeLabel = new QLabel("Type:");
	normalisedLabel = new QLabel("Normalised:");
	perInstanceLabel = new QLabel("Per Instance:");

	nameBox = new QLineEdit;
	//SIGNAL/SLOT if there's parameters

	targetBox = new QComboBox;
	targetBox->addItem("GL_ARRAY_BUFFER");
	targetBox->addItem("GL_ELEMENT_ARRAY_BUFFER");
	targetBox->addItem("GL_PIXEL_UNPACK_BUFFER");
	targetBox->connect(targetBox, SIGNAL(currentIndexChanged(int)), this, SLOT(TargetChanged(int)));

	capacityBox = new QSpinBox;
	capacityBox->setMinimum(1);
	capacityBox->setMaximum(INT_MAX);
	capacityBox->setKeyboardTracking(false);

	dataPickerBox = new QComboBox;
	dataPickerBox->addItem("Custom");
	dataPickerBox->addItem("Screen Aligned Quad Vertices");
	dataPickerBox->addItem("Screen Aligned Quad UVs");
	dataPickerBox->addItem("Cube Vertices");
	dataPickerBox->addItem("Cube UVs");
	dataPickerBox->addItem("Cube Indices");
	dataPickerBox->connect(dataPickerBox, SIGNAL(currentIndexChanged(int)), this, SLOT(DataChanged(int)));

	dataBox = new QLineEdit;
	dataBox->installEventFilter(this);

	textureBox = new QComboBox;
	for(int i = 0; i < GLSettings::TextureList.size(); ++i)
	{
		textureBox->addItem(GLSettings::TextureList.at(i)->Name());
	}
	textureBox->connect(textureBox, SIGNAL(currentIndexChanged(int)), this, SLOT(TextureChanged(int)));
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
	perInstanceBox = new QCheckBox;

	checkboxLayout->addWidget(normalisedLabel);
	checkboxLayout->addWidget(normalisedBox);
	checkboxLayout->addWidget(perInstanceLabel);
	checkboxLayout->addWidget(perInstanceBox);

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
	mainLayout->addLayout(checkboxLayout, 9, 0, 1, 2);
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
	else if(b->bName == "Cube Vertices")
	{
		dataPickerBox->setCurrentIndex(3);
		DisableBufferBoxes(true);
	}
	else if(b->bName == "Cube UVs")
	{
		dataPickerBox->setCurrentIndex(4);
		DisableBufferBoxes(true);
	}
	else if(b->bName == "Cube Indices")
	{
		dataPickerBox->setCurrentIndex(5);
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
	case 1:
	{
		dataLabel->setText("Data:");
		dataPickerBox->show();
		dataBox->show();

		capacityLabel->show();
		capacityBox->show();
		capacityBox->setValue(0);

		textureBox->hide();
		break;
	}
	case 2:
	{
		dataLabel->setText("Texture:");
		dataPickerBox->hide();
		dataBox->hide();

		capacityLabel->hide();
		capacityBox->hide();

		textureBox->show();
		textureBox->setCurrentIndex(-1);
		textureBox->setCurrentIndex(0);
		break;
	}
	}
}

void BufferPopup::TextureChanged(int i)
{
	if(i != -1)
	{
		Texture *t = GLSettings::TextureList.at(textureBox->currentIndex());
		capacityBox->setValue(t->ImageSize().width() * t->ImageSize().height() * t->FormatCount());
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
		targetBox->setCurrentIndex(0);
		capacityBox->setValue(18);
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
		targetBox->setCurrentIndex(0);
		capacityBox->setValue(12);
		dataBox->setText("");
		usageBox->setCurrentIndex(0);
		attribNameBox->setText("aUV");
		attribCapacityBox->setValue(2);
		typeBox->setCurrentIndex(0);
		normalisedBox->setChecked(false);

		DisableBufferBoxes(true);
		break;
	case 3:
		//Cube
		nameBox->setText("Cube Vertices");
		targetBox->setCurrentIndex(0);
		capacityBox->setValue(108);
		dataBox->setText("");
		usageBox->setCurrentIndex(0);
		attribNameBox->setText("aPos");
		attribCapacityBox->setValue(3);
		typeBox->setCurrentIndex(0);
		normalisedBox->setChecked(false);

		DisableBufferBoxes(true);
		break;
	case 4:
		//Cube
		nameBox->setText("Cube UVs");
		targetBox->setCurrentIndex(0);
		capacityBox->setValue(72);
		dataBox->setText("");
		usageBox->setCurrentIndex(0);
		attribNameBox->setText("aUV");
		attribCapacityBox->setValue(2);
		typeBox->setCurrentIndex(0);
		normalisedBox->setChecked(false);

		DisableBufferBoxes(true);
		break;
	case 5:
		//Cube
		nameBox->setText("Cube Indices");
		targetBox->setCurrentIndex(1);
		capacityBox->setValue(36);
		dataBox->setText("");
		usageBox->setCurrentIndex(0);
		attribNameBox->setText("aIndex");
		attribCapacityBox->setValue(1);
		typeBox->setCurrentIndex(6);
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

void BufferPopup::Save()
{
	if(Validation())
	{
		QString name = nameBox->text();
		QString target = targetBox->currentText();
		QString dataType = dataPickerBox->currentText();
		int capacity = capacityBox->value();
		QString dataPath = (targetBox->currentIndex() == 2) ? QString::number(textureBox->currentIndex()) : dataBox->text();
		QString usage = usageBox->currentText();
		QString aID = attribNameBox->text();
		int aCapacity = attribCapacityBox->value();
		QString type = typeBox->currentText();
		bool norm = normalisedBox->isChecked();
		bool perInstance = perInstanceBox->isChecked();

		if(!append)
		{
			CUGLBuffer* b;
			b = new CUGLBuffer(name, capacity, target, dataType, dataPath, usage, aID, aCapacity, type, norm, perInstance);

			if(targetBox->currentIndex() == 2)
				GLSettings::TextureList.at(textureBox->currentIndex())->PBO(b->bufID);
			GLSettings::BufferList.push_back(b);
			static_cast<GLBufferTab*>(parent())->AddToTable(b);
		}
		else
		{
			/*appBuf->bName = name;
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
			appBuf->norm = norm;*/
		}

		close();
	}
}

