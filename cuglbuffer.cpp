#include "cuglbuffer.h"

CUGLBuffer::CUGLBuffer(QString name, int capacity, std::pair<GLenum, QString> target, QString data, std::pair<GLenum, QString> usage, std::pair<QString, int> attribID, int attribCapacity, std::tuple<GLenum, QString, int> type, bool norm) : bName(name),
bCap(capacity),
bTarget(target),
bDataPath(data),
bUsage(usage),
aID(attribID),
aSize(attribCapacity),
bType(type),
norm(norm),
cuda(false),
paramID(-1)
{
	glFuncs = 0;

	GLWidget::MakeCurrent();
	glFuncs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
	if(!glFuncs)
	{
		qWarning() << "Could not obtain required OpenGL context version";
		exit(1);
	}

	bSize = bCap * std::get<2>(bType);

	if(!LoadData())
	{
		qWarning() << "Buffer data could not be loaded";
		exit(1);
	}

	if(!InitVBO())
	{
		qWarning() << "Buffer could not be initialised";
		exit(1);
	}

	GLWidget::DoneCurrent();
}

CUGLBuffer::CUGLBuffer(QString name, int capacity, std::pair<GLenum, QString> target, void *data, std::pair<GLenum, QString> usage, std::pair<QString, int> attribName, int attribCapacity, std::tuple<GLenum, QString, int> type, bool norm) : bName(name),
bCap(capacity),
bTarget(target),
bData(data),
bUsage(usage),
aID(attribName),
aSize(attribCapacity),
bType(type),
norm(norm),
cuda(false),
paramID(-1)
{
	glFuncs = 0;

	GLWidget::MakeCurrent();
	glFuncs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
	if(!glFuncs)
	{
		qWarning() << "Could not obtain required OpenGL context version";
		exit(1);
	}

	bSize = bCap * std::get<2>(bType);

	if(!InitVBO())
	{
		qWarning() << "Buffer could not be initialised";
		exit(1);
	}

	GLWidget::DoneCurrent();
}

CUGLBuffer::~CUGLBuffer()
{
	free(bData);

	UnregisterBuffer(cudaBuf);

	if(tex)
		glFuncs->glDeleteTextures(1, &tex);

	if(buf)
		glFuncs->glDeleteBuffers(1, &buf);
}

/*  Screen aligned quad
GLfloat vertices[] = {
//Vert
0.0f, 0.0f, 0.0f, 0.0f,
w, 0.0f, w, 0.0f,
w, h, w, h,

//Tex
0.0f, 0.0f, 0.0f, 0.0f,
w, h, w, h,
0.0f, h, 0.0f, h
};
*/

/************************************************************************
* target: Buffer target (GL_ARRAY_BUFFER, GL_PIXEL_UNPACK_BUFFER)
* bufferCapacity: total number of elements per buffer
* bufferData: data to be copied to buffer
* bufferUsage: usage (GL_STREAM_DRAW, GL_STATIC_DRAW, GL_DYNAMIC_DRAW...)
* attribIndex: id of attrib in shader
* attribSize: number of elements per attrib
* bufferType: type of buffer elements (GL_FLOAT...)
* normalised: T/F
************************************************************************/
bool CUGLBuffer::InitVBO()
{
	glFuncs->glGenBuffers(1, &buf);
	glFuncs->glBindBuffer(bTarget.first, buf);
	glFuncs->glBufferData(bTarget.first, bSize, bData, bUsage.first);

	cudaBuf = RegisterBuffer(buf);

	return true;
}

void CUGLBuffer::InitTex()
{
	glFuncs->glGenTextures(1, &tex);
	glFuncs->glBindTexture(GL_TEXTURE_2D, tex);

	glFuncs->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.width(), img.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);

	//glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); scale to window size?
	//glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void CUGLBuffer::Randomise(float *data, int n) {
	for(int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}

std::vector<QString> &split(const std::string &s, char delim, std::vector<QString> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(QString(item.c_str()));
    }
    return elems;
}

void CUGLBuffer::ParseFile(float *data)
{
	std::ifstream in(bDataPath.toStdString());
	std::string s = "|";
	std::vector<QString> elems;
	int counter = 0;

	getline(in, s);
	while(s.size() != 0)
	{	
		split(s, ' ', elems);
		data[counter + 0] = elems.at(0).toFloat();
		data[counter + 1] = elems.at(1).toFloat();
		data[counter + 2] = elems.at(2).toFloat();
		data[counter + 3] = elems.at(3).toFloat();
		//sscanf(s.c_str(), "%f %f %f %f", , data[counter + 1], data[counter + 2], data[counter + 3]);	//change format on type and aSize
		counter += 4;
		elems.clear();
		getline(in, s);
	}
}

bool CUGLBuffer::LoadData()
{
	switch(bTarget.first)
	{
	case GL_ARRAY_BUFFER:
	{
		bData = malloc(bSize);

		if(bDataPath == "")
			Randomise((float*)bData, bCap);
		else
			ParseFile((float*)bData);
		break;
	}
	case GL_PIXEL_UNPACK_BUFFER:
	{
		//change to texture selector or accept nullptr
		img = QImage(bDataPath);
		texSize = img.size();
		bSize = texSize.width() * texSize.height() * 4;
		img = img.convertToFormat(QImage::Format_RGB32);
		bData = (void*)img.constBits();
		InitTex();
		break;
	}
	}

	return true;
}

void CUGLBuffer::Bind()
{
	glFuncs->glBindBuffer(bTarget.first, buf);
}

void* CUGLBuffer::RegisterBuffer(GLuint buf)
{
	cudaGraphicsResource* res = 0;
	if(cudaGraphicsGLRegisterBuffer(&res, buf, cudaGraphicsRegisterFlagsNone) != cudaSuccess)
		printf("Failed to register buffer %u\n", buf);
	return res;
}

void CUGLBuffer::UnregisterBuffer(void* res)
{
	if(cudaGraphicsUnregisterResource((cudaGraphicsResource *)res) != cudaSuccess)
		puts("Failed to unregister resource for buffer");
}

void* CUGLBuffer::MapResource(void* res)
{
	if(cudaGraphicsMapResources(1, (cudaGraphicsResource **)&res) != cudaSuccess)
	{
		puts("Failed to map resource");
		return 0;
	}
	void* devPtr = 0;
	size_t size;
	if(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, (cudaGraphicsResource *)res) != cudaSuccess)
	{
		puts("Failed to get device pointer");
		return 0;
	}
	return devPtr;
}

void CUGLBuffer::UnmapResource(void* res)
{
	if(cudaGraphicsUnmapResources(1, (cudaGraphicsResource **)&res) != cudaSuccess)
		puts("Failed to unmap resource");
}
