#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_mainwindow.h"

#include<opencv2/opencv.hpp>

#include "psgan.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = Q_NULLPTR);

private:
    inline cv::Mat QImageToMat(const QImage& image);
    inline QImage MatToQImage(const cv::Mat& mat);

private slots:
    void on_openReferenceBtn_clicked();
    void on_openSourceBtn_clicked();
    void on_goBtn_clicked();

private:
    Ui::MainWindowClass ui;
    PSGAN psgan;

    cv::Mat source, reference, fake;
};
