#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QWidget>
#include <QOpenGLWidget>
#include <QOpenGLContext>
#include <QSurfaceFormat>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QKeyEvent>
#include <QTime>
#include <QString>
#include <QOpenGLTexture>

#include "cuglbuffer.h"
#include "shader.h"

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT

public:
	GLWidget(QWidget* parent = 0);
	~GLWidget();

	void initializeGL();
	void Setup();
	void BufferSetup();
	void resizeGL(int w, int h);
	void paintGL();

	static int Width();
	static int Height();
	static void MakeCurrent();
	static void DoneCurrent();
	static void Resize(int w, int h);
	static void FOV(int i);
	static void DrawMode(int i);
	static void VSync(bool b);
	static void MSAA(bool b);
	static QMatrix4x4 *ProjMatrix();
	static QMatrix4x4 *ViewMatrix();

	static std::vector<Shader*> ShaderList;

	QSize minimumSizeHint() const;
	QSize sizeHint() const;

private:
	void UpdateFPS();

	QWidget* parentWidget;
	QMatrix4x4 *projMatrix, *viewMatrix;
	GLuint drawMode;
	QTime timer;
	int width, height;
	QOpenGLFunctions_3_3_Core* glFuncs;
	QOpenGLTexture *texture;

signals:

public slots:


	// QWidget interface
protected:
	void keyPressEvent(QKeyEvent*);
};

#endif // GLWIDGET_H


