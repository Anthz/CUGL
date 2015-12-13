#pragma once
class Serializable
{
public:
	virtual ~Serializable() { }
	virtual void SaveClass() = 0;
	virtual void LoadClass() = 0;
};

