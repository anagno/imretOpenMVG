#include <iostream>
#include <QApplication>
#include <QDesktopWidget>
//#include <QPushButton>
#include "mainwindow.h"
//#include "imret/imretFuncs.hpp"    
    


int main(int argc, char *argv[]){
        //initializing
        QApplication app(argc, argv);
        //QDesktopWidget dw;
        MainWindow w;

        //  int x=dw.width()*0.5;
        //int y=dw.height()*0.5;
        //w.setFixedSize(x,y);
        w.show();
        return app.exec();
        //        return 0;
}












