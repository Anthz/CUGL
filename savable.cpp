#include "savable.h"

Savable::~Savable()
{
}

void Savable::Save(QTextStream *output, std::vector<QString> *varList)
{
	//collect certain variables from derived class
	//call base class at the end, with name of class and variables
	//save to file

	*output << varList->at(0);	//0 is the class id

	for(int i = 1; i < varList->size(); i++)
	{
		*output << "|" << varList->at(i);
	}

	endl(*output);	//end line char + flush
}
