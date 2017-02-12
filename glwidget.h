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
#include "texture.h"

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
	static void Play(bool b);
	static void StepForward();
	static void StepBackward();
	static QMatrix4x4 *ProjMatrix();
	static QMatrix4x4 *ViewMatrix();
	static void CheckFBOStatus();
	static int SetFBOTexture(GLuint id);

	static std::vector<Shader*> ShaderList;
	static std::vector<GLuint> FBOList;

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
	bool play, step, paramWarning;

signals:

public slots:


	// QWidget interface
protected:
	void keyPressEvent(QKeyEvent*);
	virtual void mousePressEvent(QMouseEvent *e) override;

};

#endif // GLWIDGET_H


