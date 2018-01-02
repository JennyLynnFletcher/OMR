#include "OMR.h"
#include "ui_mainwindow.h"
#include <score_class.h>
#include <svm.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_enter_button_clicked()
{
    QString url = ui->filepath_input->text();
    QPixmap img(url);
    ui->image->setFixedSize(img.size());
    ui->image->setPixmap(img);
    Score score;
    score_to_read = score;
    score_to_read.set_filepath(url);
    score_to_read.set_x_y(50,50);

}

void MainWindow::on_update_image_clicked()
{
    if (brightness_slider_value == -1 && contrast_slider_value == -1)
    {
        on_enter_button_clicked();
    }
    if (brightness_slider_value != ui->brightness_slider->value() || contrast_slider_value != ui->contrast_slider->value())
    {
        brightness_slider_value = ui->brightness_slider->value();
        contrast_slider_value = ui->contrast_slider->value();
        score_to_read.set_brightness_contrast(contrast_slider_value,brightness_slider_value);
        score_to_read.proccess_image();
    }
    if (original_image_selected == true)
    {
        cv::Mat original_image = score_to_read.get_original_image();
        ui->image->setPixmap(QPixmap::fromImage(QImage(original_image.data, original_image.cols, original_image.rows, original_image.step, QImage::Format_Grayscale8)));
    }
    if (binarized_image_selected == true)
    {
        cv::Mat binarized_image = score_to_read.get_binarized_image();
        ui->image->setPixmap(QPixmap::fromImage(QImage(binarized_image.data, binarized_image.cols, binarized_image.rows, binarized_image.step, QImage::Format_Grayscale8)));
    }
    if (removed_staves_selected == true)
    {
        cv::Mat removed_staves_image = score_to_read.get_removed_staves();
        ui->image->setPixmap(QPixmap::fromImage(QImage(removed_staves_image.data, removed_staves_image.cols, removed_staves_image.rows, removed_staves_image.step, QImage::Format_Grayscale8)));
    }
    if (connected_components_selected == true)
    {
        cv::Mat connected_components_image = score_to_read.get_connected_components();
        ui->image->setPixmap(QPixmap::fromImage(QImage(connected_components_image.data, connected_components_image.cols, connected_components_image.rows, connected_components_image.step, QImage::Format_RGB888)));
    }
}

void MainWindow::on_go_button_clicked()
{
    on_update_image_clicked();
    score_to_read.split_image();
    score_to_read.svm.classify_SVM();

}

void MainWindow::on_original_image_clicked()
{
    original_image_selected = true;
    binarized_image_selected = false;
    removed_staves_selected = false;
    connected_components_selected = false;
    ui->binarized_image->setChecked(false);
    ui->removed_staves->setChecked(false);
    ui->connected_components->setChecked(false);
}

void MainWindow::on_binarized_image_clicked()
{
    original_image_selected = false;
    binarized_image_selected = true;
    removed_staves_selected = false;
    connected_components_selected = false;
    ui->original_image->setChecked(false);
    ui->removed_staves->setChecked(false);
    ui->connected_components->setChecked(false);
}

void MainWindow::on_removed_staves_clicked()
{
    original_image_selected = false;
    binarized_image_selected = false;
    removed_staves_selected = true;
    connected_components_selected = false;
    ui->original_image->setChecked(false);
    ui->binarized_image->setChecked(false);
    ui->connected_components->setChecked(false);
}

void MainWindow::on_connected_components_clicked()
{
    original_image_selected = false;
    binarized_image_selected = false;
    removed_staves_selected = false;
    connected_components_selected = true;
    ui->original_image->setChecked(false);
    ui->binarized_image->setChecked(false);
    ui->removed_staves->setChecked(false);

}

void MainWindow::on_svm_train_clicked()
{
    score_to_read.svm.train_SVM();
}
