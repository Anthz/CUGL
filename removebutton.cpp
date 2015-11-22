#include "removebutton.h"

RemoveButton::RemoveButton(const QString &text) : QPushButton(text)
{
    connect(this, clicked, this, Clicked);
}

void RemoveButton::Clicked()
{
    QMessageBox::information(this, "Clicked", "Remove Clicked");
}

