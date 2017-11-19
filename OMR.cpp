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

cv::Mat MainWindow::load_mat()
{
    QString url = ui->filepath_input->text();
    QPixmap img(url);
    cv::Mat image_to_edit = cv::imread(url.toStdString());
    cv::cvtColor(image_to_edit,image_to_edit,CV_BGR2GRAY);
    return image_to_edit;
}

cv::Mat MainWindow::brightness_contrast(cv::Mat image_to_edit)
{
    float contrast = ui->contrast_slider->value();
    int brightness = ui->brightness_slider->value();
    cv::Mat new_image = cv::Mat::zeros( image_to_edit.size(), image_to_edit.type() );
    for( int y = 0; y < image_to_edit.rows; ++y)
       { for( int x = 0; x < image_to_edit.cols; ++x )
            {
            new_image.at<uchar>(y,x) =
            cv::saturate_cast<uchar>( ((contrast*3/100)+1)*(image_to_edit.at<uchar>(y,x)) + brightness);
           }
       }
    return new_image;
}

cv::Mat MainWindow::binarize(cv::Mat image)
{

    cv::adaptiveThreshold(image,image,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,41,2);
    return image;
}

int *MainWindow::histogram(cv::Mat image,int values[])
{
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            if (image.at<uchar>(y, x) == 0)
            {
                values[y]++;
            }
        }
    }
    return values;
}

std::vector<int> MainWindow::find_staves(cv::Mat new_image)
{
    int values[new_image.rows] = {0};
    *values = *histogram(new_image,values);
    std::vector<int> stave_values;
    for (int i = 0; i < new_image.rows; ++i)
    {
        if (values[i]> 0.8*new_image.cols)
        {
            stave_values.push_back(i);
        }
    }
    return stave_values;
}

cv::Mat MainWindow::remove_staves(cv::Mat image, std::vector<int> stave_values )
{
    for (int y = 0; y<(int)(stave_values.size()); ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            if (image.at<uchar>(stave_values[y]-1, x) == 255 || image.at<uchar>(stave_values[y]+1, x) == 255)
            {
                image.at<uchar>(stave_values[y], x) = 255;
            }

        }
    }
    ui->image->setPixmap(QPixmap::fromImage(QImage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888)));
    return image;
}

std::vector<cv::Mat> MainWindow::find_connected_components(cv::Mat image)
{
    cv::bitwise_not(image,image);
    cv::Mat labelImage(image.size(), CV_32S);
    int number_labels = cv::connectedComponents(image, labelImage, 8);
    std::vector<cv::Vec3b> colors(number_labels);
    colors[0] = cv::Vec3b(0, 0, 0);
    for(int label = 1; label < number_labels; ++label){
        colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    cv::Mat new_image(image.size(), CV_8UC3);
    for(int y = 0; y < new_image.rows; ++y){
        for(int x = 0; x < new_image.cols; ++x){
            int label = labelImage.at<int>(y, x);
            cv::Vec3b &pixel = new_image.at<cv::Vec3b>(y, x);
            pixel = colors[label];
        }
    }
    ui->image->setPixmap(QPixmap::fromImage(QImage(new_image.data, new_image.cols, new_image.rows, new_image.step, QImage::Format_RGB888)));
    return split_elements(image,labelImage,number_labels);
}

std::vector<cv::Mat> MainWindow::split_elements(cv::Mat image, cv::Mat label, int number_labels)
{
    image.convertTo(image,CV_8U);
    std::vector<cv::Mat> elements(number_labels);
    for (int i = 0; i<number_labels; ++i)
    {
        int max_x=0;
        int min_x=-1;
        int max_y=0;
        int min_y=-1;
        for (int y = 0; y<image.rows; ++y)
        {
            for (int x = 0; x<image.cols; ++x)
            {
                int label_present =label.at<int>(y,x);
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
        elements[i] = image(crop_area);
    }
    return elements;
}

std::vector<cv::Mat> MainWindow::standardise_elements(std::vector<cv::Mat> elements, int x, int y)
{
    std::vector<cv::Mat> standardised_elements;
    for (int i=1; i < (int)elements.size(); ++i)
    {
        cv::Mat temp_mat = elements[i];
        if (temp_mat.cols>temp_mat.rows)
        {
            float scale_factor = ceil((double)x/temp_mat.cols);
            cv::resize(temp_mat,temp_mat,cv::Size(x,y/scale_factor));
        }
        else
        {
            float scale_factor = ceil((double)y/temp_mat.rows);
            cv::resize(temp_mat,temp_mat,cv::Size(x/scale_factor,y));
        }
        cv::copyMakeBorder(temp_mat,temp_mat,0,y-temp_mat.rows,0,x-temp_mat.cols,cv::BORDER_CONSTANT);
        standardised_elements.push_back(temp_mat);
        std::string filename = "/home/jenny/Documents/Code/OMR/Elements/element" + std::to_string(i) + ".jpg";
        cv::imwrite(filename,standardised_elements[i-1]);
    }
    return standardised_elements;
}

void MainWindow::on_enter_button_clicked()
{
    QString url = ui->filepath_input->text();
    QPixmap img(url);
    ui->image->setPixmap(img);
}

void MainWindow::on_remove_button_clicked()
{
    cv::Mat image = load_mat();
    image = brightness_contrast(image);
    image = binarize(image);
    image = remove_staves(image,find_staves(image));
    std::vector<cv::Mat> elements = find_connected_components(image);
    elements = standardise_elements(elements,70,70);
}
