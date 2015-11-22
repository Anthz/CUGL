#include "controltab.h"
#include "glwidget.h"

ControlTab::ControlTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QHBoxLayout;

	first = new QPushButton("First");
	back = new QPushButton("Step Back");
	stop = new QPushButton("Stop");
	play = new QPushButton("Play");
	connect(play, SIGNAL(clicked()), this, SLOT(PlayClicked()));

	forward = new QPushButton("Step Forward");
	last = new QPushButton("Last");

	mainLayout->addWidget(first);
	mainLayout->addWidget(back);
	mainLayout->addWidget(stop);
	mainLayout->addWidget(play);
	mainLayout->addWidget(forward);
	mainLayout->addWidget(last);

	setLayout(mainLayout);
}

ControlTab::~ControlTab()
{
	delete mainLayout;

	delete first;
	delete back;
	delete stop;
	delete play;
	delete forward;
	delete last;
}

void ControlTab::PlayClicked()
{
	if(play->text() == "Play")
	{
		GLWidget::Play(true);
		play->setText("Pause");
	}
	else
	{
		GLWidget::Play(false);
		play->setText("Play");
	}
}

