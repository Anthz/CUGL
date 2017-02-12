#pragma once

#include <string>
#include <QTextStream>

class Savable
{
public:
	virtual ~Savable();
	virtual void Save(QTextStream *output, std::vector<QString> *varList);
};

