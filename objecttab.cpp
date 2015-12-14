#include "objecttab.h"
#include "glsettings.h"

ObjectTab::ObjectTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QGridLayout;
	buttonLayout = new QHBoxLayout;

	add = new QPushButton("+");
	connect(add, &QPushButton::clicked, this, &ObjectTab::Popup);

	remove = new QPushButton("-");
	connect(remove, &QPushButton::clicked, this, &ObjectTab::RemoveObject);

	buttonLayout->addWidget(add);
	buttonLayout->addWidget(remove);

	mainLayout->addLayout(buttonLayout, 0, 0);

	listModel = new QStringListModel();
	objectStringList = QStringList();

	for(int i = 0; i < GLSettings::TextureList.size(); ++i)
	{
		objectStringList.push_back(QString(GLSettings::ObjectList.at(i)->name));
	}

	listModel->setStringList(objectStringList);

	listView = new QListView;
	listView->setModel(listModel);
	connect(listView->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), this, SLOT(ObjectSelected(QItemSelection)));
	connect(listView->itemDelegate(), SIGNAL(closeEditor(QWidget*, QAbstractItemDelegate::EndEditHint)), this, SLOT(ListEditEnd(QWidget*, QAbstractItemDelegate::EndEditHint)));
	mainLayout->addWidget(listView, 1, 0);
	mainLayout->setColumnStretch(0, 1);

	ObjectPopup *o = new ObjectPopup(this);

	detailScroll = new QScrollArea;
	detailScroll->setBackgroundRole(QPalette::Dark);
	detailScroll->setWidget(o);

	mainLayout->addWidget(detailScroll, 1, 1);
	mainLayout->setColumnStretch(1, 4);

	setLayout(mainLayout);
}

ObjectTab::~ObjectTab()
{
	delete add;
	delete remove;
	delete buttonLayout;
	delete mainLayout;
}

void ObjectTab::ObjectSelected(const QItemSelection& selection)
{
	if(selection.indexes().isEmpty())
	{
		//nothing selected
		//unselect if white space is clicked in list
	}
	else
	{
		//texturePreview->setPixmap(QPixmap(QPixmap::fromImage(GLSettings::TextureList.at(selection.indexes().first().row())->Image())));
	}
}

void ObjectTab::ListEditEnd(QWidget *editor, QAbstractItemDelegate::EndEditHint)
{
	int id = listView->currentIndex().row();
	QString s = reinterpret_cast<QLineEdit*>(editor)->text();	//new name
	objectStringList.replaceInStrings(objectStringList.at(id), s);
	GLSettings::ObjectList.at(id)->name = s;
}

void ObjectTab::Popup()
{
	ObjectPopup p(this);
	p.exec();
}

QString ObjectTab::BuffersToString(std::vector<CUGLBuffer*> *buffers)
{
	QString s;
	for(int i = 0; i < buffers->size() - 1; ++i)
	{
		s.append(buffers->at(i)->bName + " | ");
	}
	s.append(buffers->at(buffers->size() - 1)->bName);

	return s;
}

void ObjectTab::TableDoubleClicked()
{
	if(!GLSettings::ObjectList.empty())
	{
		Object *obj;

		QString name = table->item(table->currentRow(), 0)->text();

		for each (Object *o in GLSettings::ObjectList)
		{
			if(o->name == name)
				obj = o;
		}

		ObjectPopup p(this, obj);
		p.exec();

		table->item(table->currentRow(), 0)->setText(obj->name);
		table->item(table->currentRow(), 1)->setText(QString::number(obj->instances));
		table->item(table->currentRow(), 2)->setText(BuffersToString(&obj->buffers));
		table->item(table->currentRow(), 3)->setText((obj->texture != nullptr) ? obj->texture->Name() : "N/A");
		table->item(table->currentRow(), 4)->setText(obj->shader->name);
	}
}

void ObjectTab::AddToTable(Object* o)
{
	if(table->rowCount() < GLSettings::ObjectList.size())
		table->insertRow(table->rowCount());
	table->setItem(table->rowCount() - 1, 0, new QTableWidgetItem(o->name));
	table->setItem(table->rowCount() - 1, 1, new QTableWidgetItem(QString::number(o->instances)));
	table->setItem(table->rowCount() - 1, 2, new QTableWidgetItem(BuffersToString(&o->buffers)));
	table->setItem(table->rowCount() - 1, 3, new QTableWidgetItem((o->texture != nullptr) ? o->texture->Name() : "N/A"));
	table->setItem(table->rowCount() - 1, 4, new QTableWidgetItem(o->shader->name));
}

void ObjectTab::RemoveObject()
{
	if(!GLSettings::ObjectList.empty() && table->currentRow() != -1)
	{
		QString name = table->item(table->currentRow(), 0)->text();

		for(int i = 0; i < GLSettings::ObjectList.size(); ++i)
		{
			if(GLSettings::ObjectList.at(i)->name == name)
			{
				GLSettings::ObjectList.erase(GLSettings::ObjectList.begin() + i);
				--i;
			}
		}

		table->removeRow(table->currentRow());

		if(table->rowCount() < 1)
		{
			table->insertRow(0);
		}
	}
}

