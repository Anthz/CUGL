#include "glbuffertab.h"
#include "glsettings.h"

GLBufferTab::GLBufferTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QVBoxLayout;
	buttonLayout = new QHBoxLayout;

	add = new QPushButton("Add");
	connect(add, &QPushButton::clicked, this, &GLBufferTab::Popup);

	remove = new QPushButton("Remove");
	connect(remove, &QPushButton::clicked, this, &GLBufferTab::RemoveBuffer);

	buttonLayout->addWidget(add);
	buttonLayout->addWidget(remove);

	mainLayout->addLayout(buttonLayout);

	table = new QTableWidget(1, 9);
	QStringList headers;
	headers << "Name" << "Capacity" << "Target" << "Data" << "Usage" << "Attribute Name" << "Attribute Capacity" << "Type" << "Normalised";
	table->setHorizontalHeaderLabels(headers);
	table->horizontalHeader()->setHighlightSections(false);
	table->verticalHeader()->setVisible(false);
	table->setEditTriggers(QAbstractItemView::NoEditTriggers);
	table->setSelectionBehavior(QAbstractItemView::SelectRows);
	table->setSelectionMode(QAbstractItemView::SingleSelection);
	connect(table, &QTableWidget::doubleClicked, this, &GLBufferTab::TableDoubleClicked);
	mainLayout->addWidget(table);
	setLayout(mainLayout);
}

GLBufferTab::~GLBufferTab()
{
 	delete mainLayout;
}

void GLBufferTab::Popup()
{
	BufferPopup p(this);
	p.exec();
}

void GLBufferTab::TableDoubleClicked()
{
	if(!GLSettings::BufferList.empty())
	{
		CUGLBuffer *buf;

		QString name = table->item(table->currentRow(), 0)->text();

		for each (CUGLBuffer *b in GLSettings::BufferList)
		{
			if(b->bName == name)
				buf = b;
		}

		BufferPopup p(this, buf);
		p.exec();

		table->item(table->rowCount() - 1, 0)->setText(buf->bName);
		table->item(table->rowCount() - 1, 1)->setText(QString::number(buf->bCap));
		table->item(table->rowCount() - 1, 2)->setText(buf->bTarget.second);
		table->item(table->rowCount() - 1, 3)->setText(buf->bDataPath);
		table->item(table->rowCount() - 1, 4)->setText(buf->bUsage.second);
		table->item(table->rowCount() - 1, 5)->setText(buf->aID.first);
		table->item(table->rowCount() - 1, 6)->setText(QString::number(buf->aSize));
		table->item(table->rowCount() - 1, 7)->setText(std::get<1>(buf->bType));
		table->item(table->rowCount() - 1, 8)->setText(QString(buf->norm ? "T" : "F"));
	}
}

void GLBufferTab::AddToTable(CUGLBuffer* b)
{
	if(table->rowCount() < GLSettings::BufferList.size())
		table->insertRow(table->rowCount());
	table->setItem(table->rowCount() - 1, 0, new QTableWidgetItem(b->bName));
	table->setItem(table->rowCount() - 1, 1, new QTableWidgetItem(QString::number(b->bCap)));
	table->setItem(table->rowCount() - 1, 2, new QTableWidgetItem(b->bTarget.second));
	table->setItem(table->rowCount() - 1, 3, new QTableWidgetItem(b->bDataPath));
	table->setItem(table->rowCount() - 1, 4, new QTableWidgetItem(b->bUsage.second));
	table->setItem(table->rowCount() - 1, 5, new QTableWidgetItem(b->aID.first));
	table->setItem(table->rowCount() - 1, 6, new QTableWidgetItem(QString::number(b->aSize)));
	table->setItem(table->rowCount() - 1, 7, new QTableWidgetItem(std::get<1>(b->bType)));
	table->setItem(table->rowCount() - 1, 8, new QTableWidgetItem(QString(b->norm ? "T" : "F")));
}

void GLBufferTab::RemoveBuffer()
{
	if(!GLSettings::BufferList.empty() && table->currentRow() != -1)
	{
		QString name = table->item(table->currentRow(), 0)->text();

		for(int i = 0; i < GLSettings::BufferList.size(); ++i)
		{
			if(GLSettings::BufferList.at(i)->bName == name)
			{
				GLSettings::BufferList.erase(GLSettings::BufferList.begin() + i);
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

