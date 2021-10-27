#pragma execution_character_set("utf-8")

#include "mainwindow.h"

#include<QDebug>
#include<QFileDialog>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    setWindowTitle("×±ÈÝÇ¨ÒÆ/·Â×±(https://github.com/EdVince)");
}

void MainWindow::on_openReferenceBtn_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("open image file"),"./", tr("Image files(*.bmp *.jpg *.png);;All files (*.*)"));
    if (!fileName.isEmpty()) {
        std::string path = fileName.toLocal8Bit().toStdString();

        reference = cv::imread(path);
        cv::cvtColor(reference, reference, cv::COLOR_BGR2RGB);

        QPixmap pixmap;
        pixmap = pixmap.fromImage(MatToQImage(reference));
        pixmap = pixmap.scaled(ui.showReference->width(), ui.showReference->height(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui.showReference->setPixmap(pixmap);
    }
}

void MainWindow::on_openSourceBtn_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("open image file"), "./", tr("Image files(*.bmp *.jpg *.png);;All files (*.*)"));
    if (!fileName.isEmpty()) {
        std::string path = fileName.toLocal8Bit().toStdString();

        source = cv::imread(path);
        cv::cvtColor(source, source, cv::COLOR_BGR2RGB);

        QPixmap pixmap;
        pixmap = pixmap.fromImage(MatToQImage(source));
        pixmap = pixmap.scaled(ui.showSource->width(), ui.showSource->height(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui.showSource->setPixmap(pixmap);
    }
}

void MainWindow::on_goBtn_clicked()
{
    if (!source.empty() && !reference.empty())
    {
        fake = psgan.go(source, reference);

        QPixmap pixmap;
        pixmap = pixmap.fromImage(MatToQImage(fake));
        pixmap = pixmap.scaled(ui.showFake->width(), ui.showFake->height(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui.showFake->setPixmap(pixmap);
    }
}

inline cv::Mat MainWindow::QImageToMat(const QImage& image)
{
    QImage swapped = image;
    if (image.format() == QImage::Format_RGB32) {
        swapped = swapped.convertToFormat(QImage::Format_RGB888);
    }

    return cv::Mat(swapped.height(), swapped.width(),
        CV_8UC3,
        const_cast<uchar*>(swapped.bits()),
        static_cast<size_t>(swapped.bytesPerLine())
    ).clone();
}

inline QImage MainWindow::MatToQImage(const cv::Mat& mat)
{
    return QImage(mat.data,
        mat.cols, mat.rows,
        static_cast<int>(mat.step),
        QImage::Format_RGB888);
}