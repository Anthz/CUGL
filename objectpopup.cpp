#include "objectpopup.h"
#include "glsettings.h"
#include "object.h"

ObjectPopup::ObjectPopup(QWidget* parent) : QDialog(parent, Qt::WindowTitleHint | Qt::WindowCloseButtonHint)
{
	append = false;
	//Object detail popup
	//setup layouts/widgets
	mainLayout = new QGridLayout;

	nameLabel = new QLabel("Name/ID:");
	instancesLabel = new QLabel("Instances:");
	bufferLabel = new QLabel("Buffers:");
	textureLabel = new QLabel("Texture:");
	shaderLabel = new QLabel("Shader:");
	fboLabel = new QLabel("FBO:");

	nameBox = new QLineEdit;

	instancesBox = new QSpinBox;
	instancesBox->setMinimum(1);
	instancesBox->setMaximum(999999);
	instancesBox->setKeyboardTracking(false);

	bufferBoxModel = new QStandardItemModel;
	for(int i = 0; i < GLSettings::BufferList.size(); i++)
	{
		QStandardItem* item = new QStandardItem();
		item->setText(GLSettings::BufferList.at(i)->bName);
		item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
		item->setData(Qt::Unchecked, Qt::CheckStateRole);
		bufferBoxModel->setItem(i, item);
		itemList.push_back(item);
	}

	bufferBox = new QComboBox;
	bufferBox->setModel(bufferBoxModel);
	connect(bufferBoxModel, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(BuffersChanged(const QModelIndex&, const QModelIndex&)));

	textureBox = new QComboBox;
	textureBox->addItem("N/A");
	for each (Texture *t in GLSettings::TextureList)
	{
		textureBox->addItem(t->Name());
	}

	shaderBox = new QComboBox;
	for each (Shader *s in GLWidget::ShaderList)
	{
		shaderBox->addItem(s->name);
	}

	fboBox = new QComboBox;
	fboBox->addItem("N/A");
	for each (GLuint i in GLWidget::FBOList)
	{
		fboBox->addItem(QString::number(i));
	}

	buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
	connect(buttons, SIGNAL(accepted()), this, SLOT(Save()));
	connect(buttons, SIGNAL(rejected()), this, SLOT(close()));

	mainLayout->addWidget(nameLabel, 0, 0);
	mainLayout->addWidget(nameBox, 0, 1);
	mainLayout->addWidget(instancesLabel, 1, 0);
	mainLayout->addWidget(instancesBox, 1, 1);
	mainLayout->addWidget(bufferLabel, 2, 0);
	mainLayout->addWidget(bufferBox, 2, 1);
	mainLayout->addWidget(textureLabel, 3, 0);
	mainLayout->addWidget(textureBox, 3, 1);
	mainLayout->addWidget(shaderLabel, 4, 0);
	mainLayout->addWidget(shaderBox, 4, 1);
	mainLayout->addWidget(fboLabel, 5, 0);
	mainLayout->addWidget(fboBox, 5, 1);
	mainLayout->addWidget(buttons, 6, 1);

	setLayout(mainLayout);
}

ObjectPopup::ObjectPopup(QWidget* parent, Object *o) : ObjectPopup(parent)
{
	append = true;
	appObj = o;

	nameBox->setText(o->name);
	instancesBox->setValue(o->instances);
	buffers = o->buffers;
	textureBox->setCurrentIndex((o->texture != nullptr) ? textureBox->findText(o->texture->Name()) : 0);
	shaderBox->setCurrentIndex(shaderBox->findText(o->shader->name));

	for each (QStandardItem *item in itemList)
	{
		for(int i = 0; i < buffers.size(); ++i)
		{
			if(item->text() == buffers.at(i)->bName)
			{
				item->setData(Qt::Checked, Qt::CheckStateRole);
			}
		}
	}

	fboBox->setCurrentIndex((o->FBO() != 0) ? fboBox->findText(QString::number(o->FBO())) : 0);
}

ObjectPopup::~ObjectPopup()
{
	delete nameLabel;
	delete instancesLabel;
	delete bufferLabel;
	delete textureLabel;
	delete shaderLabel;
	delete fboLabel;
	delete nameBox;
	delete instancesBox;

	for each (QStandardItem *i in itemList)
	{
		delete i;
	}

	delete bufferBoxModel;
	delete bufferBox;
	delete textureBox;
	delete shaderBox;
	delete fboBox;
	delete buttons;
	delete mainLayout;
}

bool ObjectPopup::Validation()
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

	if(buffers.empty())
	{
		bufferBox->setStyleSheet("border: 2px solid red");
		result = false;
	}
	else
	{
		bufferBox->setStyleSheet("");
	}

	return result;
}

//Remove buffer if deleted
//or prevent buffer deletion if tied to object
void ObjectPopup::BuffersChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight)
{
	std::cout << "Item " << topLeft.row() << " " << std::endl;
	QStandardItem* item = itemList[topLeft.row()];
	if(item->checkState() == Qt::Unchecked)
	{
		std::cout << "Unchecked!" << std::endl;

		for(int i = 0; i < buffers.size(); ++i)
		{
			if(buffers.at(i)->bName == item->text())
			{
				buffers.erase(buffers.begin() + i);
				--i;
			}
		}
	}
	else if(item->checkState() == Qt::Checked)
	{
		std::cout << "Checked!" << std::endl;
		for each(CUGLBuffer* b in GLSettings::BufferList)
		{
			if(b->bName == item->text())
			{
				bool dupe = false;

				for(int i = 0; i < buffers.size(); ++i)
				{
					if(b == buffers.at(i))
						dupe = true;
				}

				if(!dupe)
					buffers.push_back(b);
			}
		}
	}
}

void ObjectPopup::Save()
{
	if(Validation())
	{
		QString name = nameBox->text();
		int instances = instancesBox->value();

		Texture *tex = nullptr;
		if(textureBox->currentIndex() != 0)
			tex = GLSettings::TextureList.at(textureBox->currentIndex() - 1);			

		if(!append)
		{
			Object* o = new Object(name, instances, buffers, tex, GLWidget::ShaderList.at(shaderBox->currentIndex()));
			if(fboBox->currentIndex() != 0)
				o->FBO(GLWidget::FBOList.at(fboBox->currentIndex() - 1));
			GLSettings::ObjectList.push_back(o);
			static_cast<ObjectTab*>(parent())->AddToTable(o);
		}
		else
		{
			appObj->name = name;
			appObj->instances = instances;
			appObj->buffers = buffers;
			appObj->texture = tex;
			appObj->shader = GLWidget::ShaderList.at(shaderBox->currentIndex());
			if(fboBox->currentIndex() != 0)
				appObj->FBO(GLWidget::FBOList.at(fboBox->currentIndex() - 1));
		}

		close();
	}
}

