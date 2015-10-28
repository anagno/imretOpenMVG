#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QFileDialog>
#include <QString>
#include <QDebug>
#include <QRadioButton>

namespace Ui {
        class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();


private slots:
        void recognize();
        void specifyDataFolder();
        void specifyQueryFolder();
        void specifyGPSFile();
        void specifyGazeFile();
        void alg1_checked(bool checked);
        void alg2_checked(bool checked);
        void alg3_checked(bool checked);
                    
private:
    std::string gps_file_path;
    std::string gaze_file_path;
    std::string query_folder_path;
    int alg_idx;        
    Ui::MainWindow *ui;
    QPushButton *recog_button;
    QPushButton *exit_button;
    QPushButton *specify_data_button;
    QPushButton *specify_query_button;
    QPushButton *specify_gps_button;
    QPushButton *specify_gaze_button;
    QRadioButton *algorithm1_button;
    QRadioButton *algorithm2_button;
    QRadioButton *algorithm3_button;
};



#endif // MAINWINDOW_H




