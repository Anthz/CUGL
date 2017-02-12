#include "controltab.h"
#include "glwidget.h"

ControlTab::ControlTab(QWidget* parent) : QWidget(parent)
{
	mainLayout = new QHBoxLayout;

	first = new QPushButton("First");
	back = new QPushButton("Step Back");
	stop = new QPushButton("Stop");
	play = new QPushButton("Play");
	forward = new QPushButton("Step Forward");
	last = new QPushButton("Last");

	connect(back, SIGNAL(clicked()), this, SLOT(StepBackward()));
	connect(play, SIGNAL(clicked()), this, SLOT(PlayClicked()));
	connect(forward, SIGNAL(clicked()), this, SLOT(StepForward()));

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

void ControlTab::StepForward()
{
	GLWidget::StepForward();
	//update frame counter
	//implement frame counter
}

void ControlTab::StepBackward()
{
	GLWidget::StepBackward();
}

