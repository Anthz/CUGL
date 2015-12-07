#include "texture.h"

//default parameters GLenum target, GLint internalFormat, 
Texture::Texture(QString name, QString path, QImage image, int width, int height, std::pair<GLenum, QString> target, std::pair<GLint, QString> minMagFilter, std::pair<GLint, QString> wrapMode, bool fbo)
{
	glFuncs = 0;
	fboID = 0;
	//QImageReader reader(path);
	//reader.setAutoTransform(true);
	this->image = image;
	//if(image.isNull()) {
	//	qWarning() << QString("Failed to load image file %1").arg(path);
	//	return;
	//}
	
	imageSize = QSize(width, height);
	image = image.convertToFormat(QImage::Format_RGB32);
	data = image.constBits();

	this->name = name;
	this->target = target;
	this->minMagFilter = minMagFilter;
	this->wrapMode = wrapMode;
	this->fbo = fbo;

	GLWidget::MakeCurrent();
	glFuncs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
	if(!glFuncs)
	{
		qWarning() << "Could not obtain required OpenGL context version";
		exit(1);
	}
	
	glFuncs->glGenTextures(1, &texID);
	glFuncs->glBindTexture(target.first, texID);

	glFuncs->glTexParameteri(target.first, GL_TEXTURE_WRAP_S, wrapMode.first);
	glFuncs->glTexParameteri(target.first, GL_TEXTURE_WRAP_T, wrapMode.first);
	glFuncs->glTexParameteri(target.first, GL_TEXTURE_MIN_FILTER, minMagFilter.first);
	glFuncs->glTexParameteri(target.first, GL_TEXTURE_MAG_FILTER, minMagFilter.first);
	glFuncs->glTexImage2D(target.first, 0, GL_RGBA8, imageSize.width(), imageSize.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, data);

	if(fbo)
	{
		fboID = GLWidget::SetFBOTexture(texID);
	}

	glFuncs->glBindTexture(target.first, 0);

	GLWidget::DoneCurrent();
}

Texture::Texture(QString name, int width, int height, std::pair<GLenum, QString> target, std::pair<GLint, QString> minMagFilter, std::pair<GLint, QString> wrapMode, bool fbo)
{
	glFuncs = 0;
	fboID = 0;
 
// 	image = QImage();
// 	data = image.constBits();

	this->name = name;
	this->imageSize = QSize(width, height);
	this->target = target;
	this->minMagFilter = minMagFilter;
	this->wrapMode = wrapMode;
	this->fbo = fbo;

	GLWidget::MakeCurrent();
	glFuncs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
	if(!glFuncs)
	{
		qWarning() << "Could not obtain required OpenGL context version";
		exit(1);
	}

	glFuncs->glGenTextures(1, &texID);
	glFuncs->glBindTexture(target.first, texID);

	glFuncs->glTexParameteri(target.first, GL_TEXTURE_WRAP_S, wrapMode.first);
	glFuncs->glTexParameteri(target.first, GL_TEXTURE_WRAP_T, wrapMode.first);
	glFuncs->glTexParameteri(target.first, GL_TEXTURE_MIN_FILTER, minMagFilter.first);
	glFuncs->glTexParameteri(target.first, GL_TEXTURE_MAG_FILTER, minMagFilter.first);

	glFuncs->glTexImage2D(target.first, 0, GL_RGBA, imageSize.width(), imageSize.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);

	if(fbo)
	{
		fboID = GLWidget::SetFBOTexture(texID);
	}

	glFuncs->glBindTexture(target.first, 0);

	GLWidget::DoneCurrent();
}

Texture::~Texture()
{
	
}

void Texture::Bind()
{
	glFuncs->glBindTexture(target.first, texID);
}

void Texture::Unbind()
{
	glFuncs->glBindTexture(target.first, 0);
}
