#include "texturetab.h"
#include "glsettings.h"

TextureTab::TextureTab(QWidget* parent /*= 0*/)
{
	mainLayout = new QVBoxLayout;
	buttonLayout = new QHBoxLayout;
	textureLayout = new QHBoxLayout;

	add = new QPushButton("Add");
	connect(add, &QPushButton::clicked, this, &TextureTab::Popup);

	remove = new QPushButton("Remove");
	connect(remove, &QPushButton::clicked, this, &TextureTab::RemoveTexture);

	buttonLayout->addWidget(add);
	buttonLayout->addWidget(remove);

	mainLayout->addLayout(buttonLayout);

	listModel = new QStringListModel();
	textureStringList = QStringList();

	for(int i = 0; i < GLSettings::TextureList.size(); ++i)
	{
		textureStringList.push_back(QString(GLSettings::TextureList.at(i)->Name()));
	}

	listModel->setStringList(textureStringList);

	listView = new QListView;
	listView->setModel(listModel);
	listView->setSelectionMode(QAbstractItemView::ContiguousSelection);
	connect(listView->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), this, SLOT(TextureSelected(QItemSelection)));
	connect(listView->itemDelegate(), SIGNAL(closeEditor(QWidget*, QAbstractItemDelegate::EndEditHint)), this, SLOT(ListEditEnd(QWidget*, QAbstractItemDelegate::EndEditHint)));

	textureLayout->addWidget(listView);

	texturePreview = new QLabel;
	texturePreview->setBackgroundRole(QPalette::Base);
	texturePreview->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	texturePreview->setScaledContents(true);

	textureScroll = new QScrollArea;
	textureScroll->setBackgroundRole(QPalette::Dark);
	textureScroll->setWidget(texturePreview);

	textureLayout->addWidget(textureScroll);
	mainLayout->addLayout(textureLayout);

	setLayout(mainLayout);
}

TextureTab::~TextureTab()
{
	delete mainLayout;
}

void TextureTab::AddToList(Texture *t)
{
	textureStringList.push_back(t->Name());
	listModel->setStringList(textureStringList);
}

void TextureTab::TextureSelected(const QItemSelection& selection)
{
	if(selection.indexes().isEmpty())
	{
		texturePreview->setPixmap(QPixmap());
	}
	else
	{
		texturePreview->setPixmap(QPixmap(QPixmap::fromImage(GLSettings::TextureList.at(selection.indexes().first().row())->Image())));
	}

	texturePreview->adjustSize();
}

void TextureTab::ListEditEnd(QWidget *editor, QAbstractItemDelegate::EndEditHint)
{
	int id = listView->currentIndex().row();
	QString s = reinterpret_cast<QLineEdit*>(editor)->text();	//new name
	textureStringList.replaceInStrings(textureStringList.at(id), s);
	GLSettings::TextureList.at(id)->Name(s);
}

void TextureTab::RemoveTexture()
{
	if(!GLSettings::TextureList.empty() && listView->currentIndex().row() != -1)
	{

		QString name = GLSettings::TextureList.at(listView->currentIndex().row())->Name();
		bool b = GLSettings::TextureList.at(listView->currentIndex().row())->FBO();
		int fboID = -1;
		if(b)
			fboID = GLSettings::TextureList.at(listView->currentIndex().row())->FBOID();

		for(int i = 0; i < GLSettings::TextureList.size(); ++i)
		{
			if(GLSettings::TextureList.at(i)->Name() == name)
			{
				GLSettings::TextureList.erase(GLSettings::TextureList.begin() + i);
				--i;
			}
		}

		if(b)
		{
			for(int i = 0; i < GLWidget::FBOList.size(); ++i)
			{
				if(GLWidget::FBOList.at(i) == fboID)
				{
					GLWidget::FBOList.erase(GLWidget::FBOList.begin() + i);
					--i;
				}
			}
		}

		textureStringList.erase(textureStringList.begin() + listView->currentIndex().row());
		listModel->setStringList(textureStringList);
	}
}

void TextureTab::Popup()
{
	TexturePopup p(this);
	p.exec();
}
