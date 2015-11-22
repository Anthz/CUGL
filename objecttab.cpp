#include "objecttab.h"
#include "glsettings.h"

ObjectTab::ObjectTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QVBoxLayout;
	buttonLayout = new QHBoxLayout;

	add = new QPushButton("Add");
	connect(add, &QPushButton::clicked, this, &ObjectTab::Popup);

	remove = new QPushButton("Remove");
	connect(remove, &QPushButton::clicked, this, &ObjectTab::RemoveObject);

	buttonLayout->addWidget(add);
	buttonLayout->addWidget(remove);

	mainLayout->addLayout(buttonLayout);

	table = new QTableWidget(1, 5);
	QStringList headers;
	headers << "Name" << "Instances" << "Buffers" << "Texture" << "Shader";
	table->setHorizontalHeaderLabels(headers);
	table->horizontalHeader()->setHighlightSections(false);
	table->verticalHeader()->setVisible(false);
	table->setEditTriggers(QAbstractItemView::NoEditTriggers);
	table->setSelectionBehavior(QAbstractItemView::SelectRows);
	table->setSelectionMode(QAbstractItemView::SingleSelection);
	connect(table, &QTableWidget::doubleClicked, this, &ObjectTab::TableDoubleClicked);
	mainLayout->addWidget(table);
	setLayout(mainLayout);
}

ObjectTab::~ObjectTab()
{
	delete add;
	delete remove;
	delete buttonLayout;
	delete table;
	delete mainLayout;
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

