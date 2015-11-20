#include <QApplication>
#include <QDesktopWidget>
#include <QSurfaceFormat>

#include "mainwindow.h"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setSwapInterval(0); //disable vsync
	if(QCoreApplication::arguments().contains(QStringLiteral("--multisample")))
		format.setSamples(4);
	if(QCoreApplication::arguments().contains(QStringLiteral("--coreprofile")))
	{
		format.setVersion(3, 3);
		format.setProfile(QSurfaceFormat::CoreProfile);
	}
	QSurfaceFormat::setDefaultFormat(format);

	MainWindow window;
	window.resize(window.sizeHint());
	int desktopArea = QApplication::desktop()->width() * QApplication::desktop()->height();
	int widgetArea = window.width() * window.height();

	if(((float)widgetArea / (float)desktopArea) < 0.75f)
		window.show();
	else
		window.showMaximized();

	return a.exec();
}

