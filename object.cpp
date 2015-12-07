#include "object.h"

Object::Object(QString name, int instances, std::vector<CUGLBuffer*> buffers, Texture *texture, Shader *shader)
{
	glFuncs = 0;
	fbo = 0;

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
		//b->aID.second = shader->GetAttribLoc(b->aID.first);	//aID.second	//FUCKED UP MULTIPLE SHADERS!!!!!
		glFuncs->glVertexAttribPointer((GLuint)shader->GetAttribLoc(b->aID.first), b->aSize, std::get<0>(b->bType), b->norm, 0, 0);
		glFuncs->glEnableVertexAttribArray((GLuint)shader->GetAttribLoc(b->aID.first));
		if(b->Cuda())
			glFuncs->glVertexAttribDivisor((GLuint)shader->GetAttribLoc(b->aID.first), 1);

		b->Unbind();
	}

	mLoc = shader->GetUniformLoc("uModelMatrix");
	vLoc = shader->GetUniformLoc("uViewMatrix");
	pLoc = shader->GetUniformLoc("uProjMatrix");

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
		texture->Bind();

	//glFuncs->glDrawArrays(drawMode, 0, buffers.at(0)->bCap / buffers.at(0)->aSize);
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
