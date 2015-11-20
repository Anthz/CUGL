#include "addbutton.h"

AddButton::AddButton(const QString &text) : QPushButton(text)
{
    connect(this, clicked, this, Clicked);
}

void AddButton::Clicked()
{
    QMessageBox::information(this, "Clicked", "Add Clicked");
}
