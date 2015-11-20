#include "object.h"

Object::Object(QString name, int instances, std::vector<CUGLBuffer*> buffers, Texture *texture, Shader *shader)
{
	glFuncs = 0;

	this->name = name;
	this->instances = instances;
	this->buffers = buffers;
	this->texture = texture;
	this->shader = shader;

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

	for each (CUGLBuffer *b in buffers)
	{
		b->Bind();
		//FUTURE IMPROVEMENT
		//Loop for each attrib per buffer
		b->aID.second = shader->GetAttribLoc(b->aID.first);
		glFuncs->glVertexAttribPointer((GLuint)b->aID.second, b->aSize, std::get<0>(b->bType), b->norm, 0, 0);
		glFuncs->glEnableVertexAttribArray((GLuint)b->aID.second);
		if(b->bName.contains("instance", Qt::CaseInsensitive))	//add instance checkbox
			glFuncs->glVertexAttribDivisor((GLuint)b->aID.second, 1);
	}

	shader->Release();

	GLWidget::DoneCurrent();
}

Object::~Object()
{
	if(vao)
	{
		//GLWidget::MakeCurrent();
		//glFuncs->glDeleteVertexArrays(1, &vao);
		//GLWidget::DoneCurrent();
	}
}

void Object::Draw(GLenum drawMode, bool wireframe)
{
	shader->Bind();
	shader->SetUniform("uModelMatrix", modelMatrix);
	shader->SetUniform("uProjMatrix", *GLWidget::ProjMatrix());
	shader->SetUniform("uViewMatrix", *GLWidget::ViewMatrix());

	if(wireframe)
		glFuncs->glPolygonMode(GL_FRONT, GL_LINE);

	glFuncs->glBindBuffer(GL_ARRAY_BUFFER, vao);

	if(texture != nullptr)
		texture->Bind();

	//glFuncs->glDrawArrays(drawMode, 0, buffers.at(0)->bCap / buffers.at(0)->aSize);
	glFuncs->glDrawArraysInstanced(drawMode, 0, buffers.at(0)->bCap / buffers.at(0)->aSize, instances);
	texture->Unbind();
	shader->Release();
}

void Object::Move(QVector3D v)
{
	modelMatrix.translate(v);
}
