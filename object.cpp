#include "object.h"
#include "glsettings.h"

Object::Object(QString name, int instances, std::vector<int> *bufferIDs, int textureID, int shaderID)
{
	GetBuffers(bufferIDs);
	GetTexture(textureID);
	GetShader(shaderID);

	glFuncs = 0;
	fbo = 0;

	this->name = name;
	this->instances = instances;
	this->bufferIDs = *bufferIDs;
	this->textureID = textureID;
	this->shaderID = shaderID;

	modelMatrix = QMatrix4x4();
	//modelMatrix.translate(QVector3D(0.0f, 0.0f, 0.0f));

	GLWidget::MakeCurrent();
	glFuncs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
	if(!glFuncs)
	{
		qWarning() << "Could not obtain required OpenGL context version";
		exit(1);
	}

	glFuncs->glGenVertexArrays(1, &vao);
	glFuncs->glBindVertexArray(vao);

	//bind vbos in buffers to vao
	//per vbo
	//setup shader
	//find attrib location
	//store in buffer

	shader->Bind();

	int indexID = 0;
	indexed = false;

	for each (CUGLBuffer *b in buffers)
	{
		if(b->bTarget.first == GL_ELEMENT_ARRAY_BUFFER && indexed == false)
		{
			indexed = true;
			indicesID = indexID;
		}

		b->Bind();
		glFuncs->glVertexAttribPointer((GLuint)shader->GetAttribLoc(b->aID.first), b->aSize, std::get<0>(b->bType), b->norm, 0, 0);
		glFuncs->glEnableVertexAttribArray((GLuint)shader->GetAttribLoc(b->aID.first));
		if(b->PerInstance())
			glFuncs->glVertexAttribDivisor((GLuint)shader->GetAttribLoc(b->aID.first), 1);

		b->Unbind();
		indexID++;
	}

	mLoc = shader->GetUniformLoc("uModelMatrix");
	vLoc = shader->GetUniformLoc("uViewMatrix");
	pLoc = shader->GetUniformLoc("uProjMatrix");

	shader->Release();

	glFuncs->glBindVertexArray(0);

	GLWidget::DoneCurrent();
}

Object::~Object()
{
 	if(vao)
 	{
		GLWidget::MakeCurrent();
		glFuncs->glDeleteVertexArrays(1, &vao);
		GLWidget::DoneCurrent();
 	}
}

void Object::Draw(GLenum drawMode, bool wireframe)
{
	shader->Bind();

	//if matrix is used, upload
	if(mLoc != -1)
		shader->SetUniform(mLoc, modelMatrix);
	if(vLoc != -1)
		shader->SetUniform(vLoc, *GLWidget::ViewMatrix());
	if(pLoc != -1)
		shader->SetUniform(pLoc, *GLWidget::ProjMatrix());

	if(wireframe)
		glFuncs->glPolygonMode(GL_FRONT, GL_LINE);

	glFuncs->glBindVertexArray(vao);

	if(texture != nullptr)
	{
		texture->Bind();
		if(texture->PBO() != -1)
		{
			glFuncs->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, texture->PBO());
			glFuncs->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture->ImageSize().width(), texture->ImageSize().height(),
				texture->GLFmt().second, GL_UNSIGNED_BYTE, NULL);
			glFuncs->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		}
	}

	//glFuncs->glDrawArrays(drawMode, 0, buffers.at(0)->bCap / buffers.at(0)->aSize);
	if(indexed)
		glFuncs->glDrawElementsInstanced(drawMode, buffers.at(indicesID)->bCap, GL_UNSIGNED_SHORT, buffers.at(indicesID)->bData, instances);
	else
		glFuncs->glDrawArraysInstanced(drawMode, 0, buffers.at(0)->bCap / buffers.at(0)->aSize, instances);

	if(texture != nullptr)
		texture->Unbind();

	glFuncs->glBindVertexArray(0);

	shader->Release();
}

void Object::Move(QVector3D v)
{
	modelMatrix.translate(v);
}

void Object::Save(QTextStream *output, std::vector<QString> *varList)
{
	varList->push_back("o_");
	varList->push_back(name);
	varList->push_back(QString::number(instances));

	QString bufIDs;
	for(int i = 0; i < bufferIDs.size() - 1; i++)
	{
		bufIDs += QString::number(bufferIDs.at(i)) + "~";
	}
	bufIDs += QString::number(bufferIDs.at(bufferIDs.size() - 1));
	varList->push_back(bufIDs);
	varList->push_back(QString::number(textureID));
	varList->push_back(QString::number(shaderID));
	//varList->push_back(bUsage.second);	//fbo?

	Savable::Save(output, varList);
}

void Object::GetBuffers(std::vector<int> *bufferIDs)
{
	for each (int i in *bufferIDs)
	{
		buffers.push_back(GLSettings::BufferList.at(i));
	}
}

void Object::GetTexture(int texID)
{
	if(texID != -1)
		texture = GLSettings::TextureList.at(texID);
	else
		texture = nullptr;
}

void Object::GetShader(int shaderID)
{
	shader = GLWidget::ShaderList.at(shaderID);
}
