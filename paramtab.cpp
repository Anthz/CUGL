#include "paramtab.h"

ParamTab::ParamTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QVBoxLayout;
	buttonLayout = new QHBoxLayout;

	add = new QPushButton("Add");
	connect(add, &QPushButton::clicked, this, &ParamTab::AddClicked);

	remove = new QPushButton("Remove");
	connect(remove, &QPushButton::clicked, this, &ParamTab::RemoveClicked);

	buttonLayout->addWidget(add);
	buttonLayout->addWidget(remove);

	mainLayout->addLayout(buttonLayout);
	table = new QTableWidget(5, 3);
	QStringList headers;
	headers << "#" << "Name" << "Type";
	table->setHorizontalHeaderLabels(headers);
	table->horizontalHeader()->setHighlightSections(false);
	table->verticalHeader()->setVisible(false);
	mainLayout->addWidget(table);
	setLayout(mainLayout);
}

ParamTab::~ParamTab()
{
	delete buttonLayout;
	delete add;
	delete remove;
	delete table;
	delete mainLayout;
}

void ParamTab::AddClicked()
{
	QMessageBox::information(this, "Clicked", "Add Clicked");
}

void ParamTab::RemoveClicked()
{
	QMessageBox::information(this, "Clicked", "Remove Clicked");
}

