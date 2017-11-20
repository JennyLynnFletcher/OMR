#include "OMR.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <QPixmap>
#include <QtCore>
#include <cv.h>

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

void MainWindow::Score::load_mat()
{
    original_image = cv::imread(filepath.toStdString());
    cv::cvtColor(original_image,original_image,CV_BGR2GRAY);
}

void MainWindow::Score::brightness_contrast()
{
    BC_image = cv::Mat::zeros(original_image.size(), original_image.type());
    for( int y = 0; y < original_image.rows; ++y)
    { for( int x = 0; x < original_image.cols; ++x )
        {
            BC_image.at<uchar>(y,x) =
                    cv::saturate_cast<uchar>( ((contrast*3/100)+1)*(original_image.at<uchar>(y,x)) + brightness);
        }
    }
}

void MainWindow::Score::binarize()
{

    cv::adaptiveThreshold(BC_image,binarized_image,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,41,2);
}

int *MainWindow::Score::histogram(int values[])
{
    for (int y = 0; y < binarized_image.rows; ++y)
    {
        for (int x = 0; x < binarized_image.cols; ++x)
        {
            if (binarized_image.at<uchar>(y, x) == 0)
            {
                values[y]++;
            }
        }
    }
    return values;
}

void MainWindow::Score::find_staves()
{
    int values[binarized_image.rows] = {0};
    *values = *histogram(values);
    std::vector<int> stave_values;
    for (int i = 0; i < binarized_image.rows; ++i)
    {
        if (values[i]> 0.8*binarized_image.cols)
        {
            stave_values.push_back(i);
        }
    }
    staves = stave_values;
}

void MainWindow::Score::remove_staves()
{
    removed_staves = binarized_image;
    for (int y = 0; y<(int)(staves.size()); ++y)
    {
        for (int x = 0; x < removed_staves.cols; ++x)
        {
            if (removed_staves.at<uchar>(staves[y]-1, x) == 255 || removed_staves.at<uchar>(staves[y]+1, x) == 255)
            {
                removed_staves.at<uchar>(staves[y], x) = 255;
            }

        }
    }
}

void MainWindow::Score::find_connected_components()
{
    cv::bitwise_not(removed_staves,removed_staves);
    cv::Mat temp(removed_staves.size(), CV_32S);
    label_image = temp;
    number_labels = cv::connectedComponents(removed_staves, label_image, 8);
    std::vector<cv::Vec3b> colors(number_labels);
    colors[0] = cv::Vec3b(0, 0, 0);
    for(int label = 1; label < number_labels; ++label){
        colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    cv::Mat mat(removed_staves.size(), CV_8UC3);
    coloured_connected_components = mat;
    for(int y = 0; y < coloured_connected_components.rows; ++y){
        for(int x = 0; x < coloured_connected_components.cols; ++x){
            int label = label_image.at<int>(y, x);
            cv::Vec3b &pixel = coloured_connected_components.at<cv::Vec3b>(y, x);
            pixel = colors[label];
        }
    }
}

void MainWindow::Score::split_elements()
{
    binarized_image.convertTo(removed_staves,CV_8U);
    std::vector<cv::Mat> elements(number_labels);
    for (int i = 0; i<number_labels; ++i)
    {
        int max_x=0;
        int min_x=-1;
        int max_y=0;
        int min_y=-1;
        for (int y = 0; y<binarized_image.rows; ++y)
        {
            for (int x = 0; x<binarized_image.cols; ++x)
            {
                int label_present =label_image.at<int>(y,x);
                if (label_present == i)
                {
                    if (x > max_x)
                    {
                        max_x = x;
                    }
                    if (min_x > x || min_x == -1)
                    {
                        min_x = x;
                    }
                    if (y > max_y)
                    {
                        max_y = y;
                    }
                    if (min_x > x || min_y == -1)
                    {
                        min_y = y;
                    }
                }
            }
        }
        cv::Rect crop_area;
        crop_area.x = min_x;
        crop_area.y = min_y;
        crop_area.width = max_x - min_x;
        crop_area.height = max_y - min_y;
        elements[i] = binarized_image(crop_area);
    }
}

void MainWindow::Score::standardise_elements()
{
    for (int i=1; i < (int)elements.size(); ++i)
    {
        cv::Mat temp_mat = elements[i];
        if (temp_mat.cols>temp_mat.rows)
        {
            float scale_factor = ceil((double)output_x/temp_mat.cols);
            cv::resize(temp_mat,temp_mat,cv::Size(output_x,output_y/scale_factor));
        }
        else
        {
            float scale_factor = ceil((double)output_y/temp_mat.rows);
            cv::resize(temp_mat,temp_mat,cv::Size(output_x/scale_factor,output_y));
        }
        cv::copyMakeBorder(temp_mat,temp_mat,0,output_y-temp_mat.rows,0,output_x-temp_mat.cols,cv::BORDER_CONSTANT);
        standardised_elements.push_back(temp_mat);
        std::string filename = "/home/jenny/Documents/Code/OMR/Elements/element" + std::to_string(i) + ".jpg";
        cv::imwrite(filename,standardised_elements[i-1]);
    }
}

std::vector<int> MainWindow::Score::get_staves()
{
    return staves;
}

cv::Mat MainWindow::Score::get_original_image()
{
    return original_image;
}

cv::Mat MainWindow::Score::get_binarized_image()
{
    return binarized_image;
}

cv::Mat MainWindow::Score::get_removed_staves()
{
    return removed_staves;
}

cv::Mat MainWindow::Score::get_connected_components()
{
    return coloured_connected_components;
}

std::vector<cv::Mat> MainWindow::Score::get_elements()
{
    return elements;
}

std::vector<cv::Mat> MainWindow::Score::get_standardised_elements()
{
    return standardised_elements;
}

void MainWindow::Score::set_x_y(int x, int y)
{
    output_x = x;
    output_y = y;
}

void MainWindow::Score::set_filepath(QString input_filepath)
{
    filepath = input_filepath;
}

void MainWindow::Score::set_brightness_contrast(float input_contrast, int input_brightness)
{
    contrast = input_contrast;
    brightness = input_brightness;
}

void MainWindow::Score::proccess_image()
{
    load_mat();
    brightness_contrast();
    binarize();
    find_staves();
    remove_staves();
    find_connected_components();
}


void MainWindow::on_enter_button_clicked()
{
    QString url = ui->filepath_input->text();
    QPixmap img(url);
    ui->image->setFixedSize(img.size());
    ui->image->setPixmap(img);
    MainWindow::Score score;
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
        cv::Mat new_image = score_to_read.get_original_image();
        //new_image.convertTo(new_image,CV_8UC3);
        ui->image->setPixmap(QPixmap::fromImage(QImage(new_image.data, new_image.cols, new_image.rows, new_image.step, QImage::Format_Grayscale8)));
    }
    if (binarized_image_selected == true)
    {
        cv::Mat new_image = score_to_read.get_binarized_image();
        //new_image.convertTo(new_image,CV_8UC2);
        ui->image->setPixmap(QPixmap::fromImage(QImage(new_image.data, new_image.cols, new_image.rows, new_image.step, QImage::Format_Grayscale8)));
    }
    if (removed_staves_selected == true)
    {
        cv::Mat new_image = score_to_read.get_removed_staves();
        //new_image.convertTo(new_image,CV_8UC2);
        ui->image->setPixmap(QPixmap::fromImage(QImage(new_image.data, new_image.cols, new_image.rows, new_image.step, QImage::Format_Grayscale8)));
    }
    if (connected_components_selected == true)
    {
        cv::Mat new_image = score_to_read.get_connected_components();
        //new_image.convertTo(new_image,CV_8UC3);
        ui->image->setPixmap(QPixmap::fromImage(QImage(new_image.data, new_image.cols, new_image.rows, new_image.step, QImage::Format_RGB888)));
    }
}

void MainWindow::on_go_button_clicked()
{

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
