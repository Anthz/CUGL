#include "texture.h"
#include "glsettings.h"

Texture::Texture(QString name, QString path, int width, int height, QImage::Format fmt, QString targetString, QString filterString, QString wrapString, bool fbo, int pbo)
{
	GetGLTarget(targetString);
	GetGLMinMagFilter(filterString);
	GetGLWrapMode(wrapString);

	glFuncs = 0;
	fboID = 0;

	if(path != "")
	{
		image = QImage(path);
		image = image.convertToFormat(fmt);
	}
	else
	{
		image = QImage(width, height, fmt);
	}
	data = image.constBits();

	this->name = name;
	this->imageSize = QSize(width, height);
	this->dataPath = path;
	this->target = target;
	this->minMagFilter = minMagFilter;
	this->wrapMode = wrapMode;
	this->fbo = fbo;
	this->pbo = pbo;
	GetGLFormat();	//after image has been converted

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
	glFuncs->glGenerateMipmap(target.first);

	glFuncs->glTexImage2D(target.first, 0, glFmt.first, imageSize.width(), imageSize.height(), 0, glFmt.second, GL_UNSIGNED_BYTE, data);

	if(fbo)
	{
		fboID = GLWidget::SetFBOTexture(texID);
	}

	glFuncs->glBindTexture(target.first, 0);

	GLWidget::DoneCurrent();
}

Texture::~Texture()
{
	glFuncs->glDeleteTextures(1, &texID);
}

void Texture::Bind()
{
	glFuncs->glBindTexture(target.first, texID);
}

void Texture::Unbind()
{
	glFuncs->glBindTexture(target.first, 0);
}

unsigned int Texture::FormatCount()
{
	switch(image.format())
	{
	case QImage::Format_ARGB32:
		return 4;
		break;
	case QImage::Format_RGB888:
		return 3;
		break;
	case QImage::Format_Grayscale8:
		return 1;
		break;
	}
}

void Texture::GetGLFormat()
{
	switch(image.format())
	{
	case QImage::Format_ARGB32:
		glFmt.first = GL_RGBA8;
		glFmt.second = GL_BGRA;
		break;
	case QImage::Format_RGB888:
		glFmt.first = GL_RGB8;
		glFmt.second = GL_BGRA;	//use over GL_RGB?
		break;
	case QImage::Format_Grayscale8:
		glFmt.first = GL_RED;
		glFmt.second = GL_RED;
		break;
	}
}

void Texture::GetGLTarget(QString targetString)
{
	if(targetString.contains("1D"))
		target.first = GL_TEXTURE_1D;

	else if(targetString.contains("2D"))
		target.first = GL_TEXTURE_2D;

	else if(targetString.contains("3D"))
		target.first = GL_TEXTURE_3D;

	else if(targetString.contains("RECTANGLE"))
		target.first = GL_TEXTURE_RECTANGLE;

	else if(targetString.contains("CUBE_MAP"))
		target.first = GL_TEXTURE_CUBE_MAP;

	target.second = targetString;
}

void Texture::GetGLMinMagFilter(QString filterString)
{
	if(filterString == "GL_NEAREST")
		minMagFilter.first = GL_NEAREST;

	else if(filterString == "GL_LINEAR")
		minMagFilter.first = GL_LINEAR;

	else if(filterString == "GL_NEAREST_MIPMAP_NEAREST")
		minMagFilter.first = GL_NEAREST_MIPMAP_NEAREST;

	else if(filterString == "GL_LINEAR_MIPMAP_NEAREST")
		minMagFilter.first = GL_LINEAR_MIPMAP_NEAREST;

	else if(filterString == "GL_NEAREST_MIPMAP_LINEAR")
		minMagFilter.first = GL_NEAREST_MIPMAP_LINEAR;

	else if(filterString == "GL_LINEAR_MIPMAP_LINEAR")
		minMagFilter.first = GL_LINEAR_MIPMAP_LINEAR;

	minMagFilter.second = filterString;
}

void Texture::GetGLWrapMode(QString wrapString)
{
	if(wrapString == "GL_REPEAT")
		wrapMode.first = GL_REPEAT;

	else if(wrapString == "GL_MIRRORED_REPEAT")
		wrapMode.first = GL_MIRRORED_REPEAT;

	else if(wrapString == "GL_CLAMP_TO_EDGE")
		wrapMode.first = GL_CLAMP_TO_EDGE;

	else if(wrapString == "GL_CLAMP_TO_BORDER")
		wrapMode.first = GL_CLAMP_TO_BORDER;

	else if(wrapString == "GL_MIRROR_CLAMP_TO_EDGE")
		wrapMode.first = GL_MIRROR_CLAMP_TO_EDGE;

	wrapMode.second = wrapString;
}

void Texture::Save(QTextStream *output, std::vector<QString> *varList)
{
	varList->push_back("t_");
	varList->push_back(name);
	varList->push_back(dataPath);
	varList->push_back(QString::number(imageSize.width()));
	varList->push_back(QString::number(imageSize.height()));
	varList->push_back(QString::number(image.format()));
	varList->push_back(target.second);
	varList->push_back(minMagFilter.second);
	varList->push_back(wrapMode.second);
	varList->push_back(QString::number(fbo));
	varList->push_back(QString::number(pbo));

	Savable::Save(output, varList);
}
