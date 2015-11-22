#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMessageBox>
#include <QOpenGLTexture>
#include <QListView>
#include <QStringListModel>
#include <QLabel>
#include <QFileDialog>
#include <QScrollArea>

#include "texture.h"

class TextureTab : public QWidget
{
	Q_OBJECT

public:
	explicit TextureTab(QWidget* parent = 0);
	~TextureTab();

	void AddToList(Texture *t);	

private:
	QVBoxLayout* mainLayout;
	QHBoxLayout* buttonLayout;
	QHBoxLayout* textureLayout;
	QPushButton* add;
	QPushButton* remove;
	QListView *listView;
	QStringListModel *listModel;
	QLabel *texturePreview;
	QScrollArea *textureScroll;
	QStringList textureStringList;
signals:

public slots:

private slots:
	void TextureSelected(const QItemSelection& selection);
	void ListEditEnd(QWidget*, QAbstractItemDelegate::EndEditHint);
	void RemoveTexture();
	void Popup();
};

