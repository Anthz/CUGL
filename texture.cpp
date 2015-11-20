#include "texture.h"

//default parameters GLenum target, GLint internalFormat, 
Texture::Texture(QString name, QString path, GLenum target, GLint minMagFilter, GLint wrapMode)
{
	glFuncs = 0;

	QImageReader reader(path);
	reader.setAutoTransform(true);
	image = reader.read();
	if(image.isNull()) {
		qWarning() << QString("Failed to load image file %1").arg(path);
		return;
	}

	imageSize = image.size();
	image = image.convertToFormat(QImage::Format_RGB32);
	data = image.constBits();

	this->name = name;
	this->target = target;
	this->minMagFilter = minMagFilter;
	this->wrapMode = wrapMode;

	GLWidget::MakeCurrent();
	glFuncs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
	if(!glFuncs)
	{
		qWarning() << "Could not obtain required OpenGL context version";
		exit(1);
	}
	
	glFuncs->glGenTextures(1, &texID);
	glFuncs->glBindTexture(target, texID);

	glFuncs->glTexParameteri(target, GL_TEXTURE_WRAP_S, wrapMode);
	glFuncs->glTexParameteri(target, GL_TEXTURE_WRAP_T, wrapMode);
	glFuncs->glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minMagFilter);
	glFuncs->glTexParameteri(target, GL_TEXTURE_MAG_FILTER, minMagFilter);

	glFuncs->glTexImage2D(target, 0, GL_RGBA8, image.width(), image.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, data);

	glFuncs->glBindTexture(target, 0);

	GLWidget::DoneCurrent();
}

Texture::~Texture()
{
	
}

void Texture::Bind()
{
	glFuncs->glBindTexture(target, texID);
}

void Texture::Unbind()
{
	glFuncs->glBindTexture(target, 0);
}
