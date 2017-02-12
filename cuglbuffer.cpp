#include "cuglbuffer.h"
#include "glsettings.h"

#pragma region Shapes
float quad[] =
{
	-0.5f, -0.5f, 0.0f,
	-0.5f, 0.5f, 0.0f,
	0.5f, 0.5f, 0.0f,
	0.5f, 0.5f, 0.0f,
	0.5f, -0.5f, 0.0f,
	-0.5f, -0.5f, 0.0f
};

float quadUV[] =
{
	0.0f, 1.0f,
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f
};

float triangle[] =
{
	-1.0f, -1.0f, 0.0f,
	0.0f, 1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
};

float cube_vertices[] = {
	// front
	-1.0, -1.0, 1.0,
	1.0, -1.0, 1.0,
	1.0, 1.0, 1.0,
	1.0, 1.0, 1.0,
	-1.0, 1.0, 1.0,
	-1.0, -1.0, 1.0,
	// top
	-1.0, 1.0, 1.0,
	1.0, 1.0, 1.0,
	1.0, 1.0, -1.0,
	1.0, 1.0, -1.0,
	-1.0, 1.0, -1.0,
	-1.0, 1.0, 1.0,
	// back
	1.0, -1.0, -1.0,
	-1.0, -1.0, -1.0,
	-1.0, 1.0, -1.0,
	-1.0, 1.0, -1.0,
	1.0, 1.0, -1.0,
	1.0, -1.0, -1.0,
	// bottom
	-1.0, -1.0, -1.0,
	1.0, -1.0, -1.0,
	1.0, -1.0, 1.0,
	1.0, -1.0, 1.0,
	-1.0, -1.0, 1.0,
	-1.0, -1.0, -1.0,
	// left
	-1.0, -1.0, -1.0,
	-1.0, -1.0, 1.0,
	-1.0, 1.0, 1.0,
	-1.0, 1.0, 1.0,
	-1.0, 1.0, -1.0,
	-1.0, -1.0, -1.0,
	// right
	1.0, -1.0, 1.0,
	1.0, -1.0, -1.0,
	1.0, 1.0, -1.0,
	1.0, 1.0, -1.0,
	1.0, 1.0, 1.0,
	1.0, -1.0, 1.0,
};

float cube_texcoords[] = {
	// front
	0.0, 0.0,
	1.0, 0.0,
	1.0, 1.0,
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0,
	// top
	0.0, 0.0,
	1.0, 0.0,
	1.0, 1.0,
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0,
	// back
	0.0, 0.0,
	1.0, 0.0,
	1.0, 1.0,
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0,
	// bottom
	0.0, 0.0,
	1.0, 0.0,
	1.0, 1.0,
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0,
	// left
	0.0, 0.0,
	1.0, 0.0,
	1.0, 1.0,
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0,
	// right
	0.0, 0.0,
	1.0, 0.0,
	1.0, 1.0,
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0,
};

GLushort cube_elements[] = {
	// front
	0, 1, 2,
	2, 3, 0,
	// top
	4, 5, 6,
	6, 7, 4,
	// back
	8, 9, 10,
	10, 11, 8,
	// bottom
	12, 13, 14,
	14, 15, 12,
	// left
	16, 17, 18,
	18, 19, 16,
	// right
	20, 21, 22,
	22, 23, 20,
};

#pragma endregion Shapes

CUGLBuffer::CUGLBuffer(QString name, int capacity, QString target, QString data, QString dataPath, QString usage, QString attribID, int attribCapacity, QString type, bool norm, bool perInst)
{
	GetGLTarget(target);
	GetGLUsage(usage);
	aID = std::make_pair(attribID, -1);	//refactor?
	GetGLType(type);

	glFuncs = 0;

	bName = name;
	bCap = capacity;
	bDataType = data;	//e.g. SAQ/Custom etc (not var type!)
	bDataPath = dataPath;
	aSize = attribCapacity;
	norm = norm;
	cuda = false;
	perInstance = perInst;
	paramID = -1;

	GLWidget::MakeCurrent();
	glFuncs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
	if(!glFuncs)
	{
		qWarning() << "Could not obtain required OpenGL context version";
		exit(1);
	}

	bSize = bCap * std::get<2>(bType);

	if(!LoadData(data))
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

CUGLBuffer::~CUGLBuffer()
{
	free(bData);

	UnregisterBuffer(cudaBuf);

	if(tex)
		glFuncs->glDeleteTextures(1, &tex);

	if(bufID)
		glFuncs->glDeleteBuffers(1, &bufID);
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
	glFuncs->glGenBuffers(1, &bufID);
	glFuncs->glBindBuffer(bTarget.first, bufID);
	glFuncs->glBufferData(bTarget.first, bSize, bData, bUsage.first);

	cudaBuf = RegisterBuffer(bufID);

	return true;
}

std::vector<QString> &split(const std::string &s, char delim, std::vector<QString> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(QString(item.c_str()));
    }
    return elems;
}

//implement custom parser
bool CUGLBuffer::LoadData(QString dataType)
{
	switch(bTarget.first)
	{
		case GL_ARRAY_BUFFER:
		{
			bData = malloc(bSize);	//redundant when assigning to array?

			if(dataType != "Custom")
			{
				if(dataType.contains("Cube"))
				{
					if(dataType.contains("Vertices"))
						bData = cube_vertices;
					else if(dataType.contains("UVs"))
						bData = cube_texcoords;
				}
				if(dataType.contains("Quad"))
				{
					if(dataType.contains("Vertices"))
						bData = quad;
					else if(dataType.contains("UVs"))
						bData = quadUV;
				}
			}
			else
			{
				if(bDataPath == "")
					RandomData<int>((int*)bData, bCap, 0.5, 1.5);
				else
					ParseFile<byte>((byte*)bData);
			}
			break;
		}
		case GL_ELEMENT_ARRAY_BUFFER:
		{
			if(dataType.contains("Cube"))
				bData = cube_elements;
			break;
		}

		case GL_PIXEL_UNPACK_BUFFER:
		{
			bData = (void*)GLSettings::TextureList.at(bDataPath.toInt())->Data();
			break;
		}
	}

	return true;
}

void CUGLBuffer::GetGLTarget(QString targetString)
{
	if(targetString == "GL_ARRAY_BUFFER")
	{
		bTarget.first = GL_ARRAY_BUFFER;
	}

	if(targetString == "GL_ELEMENT_ARRAY_BUFFER")
	{
		bTarget.first = GL_ELEMENT_ARRAY_BUFFER;
	}

	if(targetString == "GL_PIXEL_UNPACK_BUFFER")
	{
		bTarget.first = GL_PIXEL_UNPACK_BUFFER;
	}

	bTarget.second = targetString;
}

void CUGLBuffer::GetGLUsage(QString usageString)
{
	if(usageString == "GL_DYNAMIC_DRAW")
	{
		bUsage.first = GL_DYNAMIC_DRAW;
	}

	if(usageString == "GL_DYNAMIC_COPY")
	{
		bUsage.first = GL_DYNAMIC_COPY;
	}

	if(usageString == "GL_STATIC_DRAW")
	{
		bUsage.first = GL_STATIC_DRAW;
	}

	if(usageString == "GL_STATIC_COPY")
	{
		bUsage.first = GL_STATIC_COPY;
	}

	bUsage.second = usageString;
}

void CUGLBuffer::GetAttribID(QString attribString)
{

}

void CUGLBuffer::GetGLType(QString typeString)
{
	GLenum typeEnum;
	int size;

	if(typeString == "GL_FLOAT")
	{
		typeEnum = GL_FLOAT;
		size = sizeof(float);
	}

	if(typeString == "GL_HALF_FLOAT")
	{
		typeEnum = GL_HALF_FLOAT;
		size = sizeof(float) / 2;
	}

	if(typeString == "GL_DOUBLE")
	{
		typeEnum = GL_DOUBLE;
		size = sizeof(double);
	}

	if(typeString == "GL_INT")
	{
		typeEnum = GL_INT;
		size = sizeof(int);
	}

	if(typeString == "GL_UNSIGNED_INT")
	{
		typeEnum = GL_UNSIGNED_INT;
		size = sizeof(unsigned int);
	}

	if(typeString == "GL_SHORT")
	{
		typeEnum = GL_SHORT;
		size = sizeof(short);
	}

	if(typeString == "GL_UNSIGNED_SHORT")
	{
		typeEnum = GL_UNSIGNED_SHORT;
		size = sizeof(unsigned short);
	}

	if(typeString == "GL_BYTE")
	{
		typeEnum = GL_BYTE;
		size = sizeof(byte);
	}

	if(typeString == "GL_UNSIGNED_BYTE")
	{
		typeEnum = GL_UNSIGNED_BYTE;
		size = sizeof(byte);
	}

	bType = std::make_tuple(typeEnum, typeString, size);
}

void CUGLBuffer::Save(QTextStream *output, std::vector<QString> *varList)
{
	varList->push_back("b_");
	varList->push_back(bName);
	varList->push_back(QString::number(bCap));
	varList->push_back(bTarget.second);
	varList->push_back(bDataType);
	varList->push_back(bDataPath);
	varList->push_back(bUsage.second);
	varList->push_back(aID.first);
	varList->push_back(QString::number(aSize));
	varList->push_back(std::get<1>(bType));
	varList->push_back(QString::number(norm));
	varList->push_back(QString::number(perInstance));

	Savable::Save(output, varList);
}

void CUGLBuffer::Bind()
{
	glFuncs->glBindBuffer(bTarget.first, bufID);
}

void CUGLBuffer::Unbind()
{
	glFuncs->glBindBuffer(bTarget.first, 0);
}

void* CUGLBuffer::RegisterBuffer(GLuint buf)
{
	cudaGraphicsResource* res = 0;
	ERRORCHECK(cudaGraphicsGLRegisterBuffer(&res, buf, cudaGraphicsRegisterFlagsNone));
	return res;
}

void CUGLBuffer::UnregisterBuffer(void* res)
{
	ERRORCHECK(cudaGraphicsUnregisterResource((cudaGraphicsResource *)res));
}

//Back off Open-GL
void* CUGLBuffer::MapResource(void* res)
{
	void* devPtr = 0;
	size_t size;
	ERRORCHECK(cudaGraphicsMapResources(1, (cudaGraphicsResource **)&res));
	ERRORCHECK(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, (cudaGraphicsResource *)res));
	return devPtr;
}

void CUGLBuffer::UnmapResource(void* res)
{
	ERRORCHECK(cudaGraphicsUnmapResources(1, (cudaGraphicsResource **)&res));
}

