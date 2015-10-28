#include "mainwindow.h"
#include "ui_mainwindow.h"
//#include "Recognizer.h"
#include "imret/imretFuncs.hpp"

MainWindow::MainWindow(QWidget *parent): QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);
    
    //recognizing
    recog_button = new QPushButton("Recognize", this);
    recog_button->setGeometry(QRect(QPoint(30, 200), QSize(100, 40)));
    recog_button->show();
    connect(recog_button, SIGNAL(clicked()), this, SLOT(recognize()));

    //exit
    exit_button = new QPushButton("Exit", this);
    exit_button->setGeometry(QRect(QPoint(270, 200), QSize(100, 40)));
    exit_button->show();
    connect(exit_button, SIGNAL(clicked()), this, SLOT(close()));

    //specify data folder
    // specify_data_button = new QPushButton("specify data folder", this);
    // specify_data_button->setGeometry(QRect(QPoint(30, 100), QSize(150, 40)));
    // specify_data_button->show();
    // connect(specify_data_button, SIGNAL(clicked()), this, SLOT(specifyDataFolder()));

    //specify query folder
    specify_query_button = new QPushButton("specify query folder", this);
    specify_query_button->setGeometry(QRect(QPoint(30, 10), QSize(150, 40)));
    specify_query_button->show();
    connect(specify_query_button, SIGNAL(clicked()), this, SLOT(specifyQueryFolder()));

    //specify GPS file
    specify_gps_button = new QPushButton("specify GPS file", this);
    specify_gps_button->setGeometry(QRect(QPoint(270, 10), QSize(150, 40)));
    specify_gps_button->show();
    connect(specify_gps_button, SIGNAL(clicked()), this, SLOT(specifyGPSFile()));

     //specify gaze file
    specify_gaze_button = new QPushButton("specify gaze file", this);
    specify_gaze_button->setGeometry(QRect(QPoint(500, 10), QSize(150, 40)));
    specify_gaze_button->show();
    connect(specify_gaze_button, SIGNAL(clicked()), this, SLOT(specifyGazeFile()));

    //radiobutton alg1
    algorithm1_button = new QRadioButton("Algorithm 3", this);
    algorithm1_button->setGeometry(QRect(QPoint(30, 100), QSize(150, 40)));
    connect(algorithm1_button,SIGNAL(toggled(bool)),this,SLOT(alg1_checked(bool)));
    //radiobutton alg2
    algorithm2_button = new QRadioButton("Algorithm 2+3", this);
    algorithm2_button->setGeometry(QRect(QPoint(270, 100), QSize(150, 40)));
    connect(algorithm2_button,SIGNAL(toggled(bool)),this,SLOT(alg2_checked(bool)));
    //radiobutton alg3
    algorithm3_button = new QRadioButton("Algorithm 1+2+3", this);
    algorithm3_button->setGeometry(QRect(QPoint(500, 100), QSize(150, 40)));
    connect(algorithm3_button,SIGNAL(toggled(bool)),this,SLOT(alg3_checked(bool)));
    
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::recognize()
{
        
        compute(alg_idx, gps_file_path, gaze_file_path, query_folder_path);
}


void MainWindow::specifyDataFolder(){
        QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                "../workspace",
                                                QFileDialog::ShowDirsOnly
                                                | QFileDialog::DontResolveSymlinks);

        qDebug().nospace()<<qPrintable(dir) ;
}

void MainWindow::specifyQueryFolder(){
        QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                "../workspace",
                                                QFileDialog::ShowDirsOnly
                                                | QFileDialog::DontResolveSymlinks);

        //        qDebug().nospace()<<qPrintable(dir) ;
        query_folder_path = dir.toUtf8().constData();
        std::cout << query_folder_path << std::endl;
}


void MainWindow::specifyGPSFile(){
        QString fileName = QFileDialog::getOpenFileName(this,
                                                tr("Open GPS"), "../workspace/GPS", tr("text Files (*.txt)"));
        //        qDebug().nospace() << qPrintable(fileName);
        gps_file_path = fileName.toUtf8().constData();
        std::cout << gps_file_path << std::endl;
}


void MainWindow::specifyGazeFile(){
        QString fileName = QFileDialog::getOpenFileName(this,
                                                tr("Open Gazes"), "../workspace/gazes", tr("text Files (*.txt)"));
        //qDebug().nospace() << qPrintable(fileName);
        gaze_file_path = fileName.toUtf8().constData();
        std::cout << gaze_file_path << std::endl;

}



void MainWindow::alg1_checked(bool checked){
        if(checked){
                alg_idx = 1;
                std::cout << "algorithm 3" << std::endl;
        }
}

void MainWindow::alg2_checked(bool checked){

        if(checked){
                alg_idx = 2;
                std::cout << "algorithm 2+3" << std::endl;
        }

}

void MainWindow::alg3_checked(bool checked){
        if(checked){
                alg_idx = 3;
                std::cout << "algorithm 1+2+3" << std::endl;
        }
}

