#include "OMR.h"
#include "ui_mainwindow.h"
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

void Score::load_mat()
{
    original_image = cv::imread(filepath.toStdString());
    if (original_image.empty()== false)
    {
        cv::cvtColor(original_image,original_image,CV_BGR2GRAY);
        image_exists = true;
    }
    else
    {
        image_exists = false;
    }

}

void Score::brightness_contrast()
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

void Score::binarize()
{

    cv::adaptiveThreshold(BC_image,binarized_image,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,41,2);
}

int *Score::histogram(int values[])
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

void Score::find_staves()
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

void Score::remove_staves()
{
    binarized_image.copyTo(removed_staves);
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

void Score::find_connected_components()
{
    cv::bitwise_not(removed_staves,removed_staves);
    cv::Mat label_image_empty(removed_staves.size(), CV_32S);
    label_image = label_image_empty;
    number_labels = cv::connectedComponents(removed_staves, label_image, 8);
    std::vector<cv::Vec3b> colors(number_labels);
    colors[0] = cv::Vec3b(0, 0, 0);
    for(int label = 1; label < number_labels; ++label){
        colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    cv::Mat coloured_connected_components_empty(removed_staves.size(), CV_8UC3);
    coloured_connected_components = coloured_connected_components_empty;
    for(int y = 0; y < coloured_connected_components.rows; ++y){
        for(int x = 0; x < coloured_connected_components.cols; ++x){
            int label = label_image.at<int>(y, x);
            cv::Vec3b &pixel = coloured_connected_components.at<cv::Vec3b>(y, x);
            pixel = colors[label];
        }
    }
}

void Score::split_elements()
{
    elements.clear();
    removed_staves.convertTo(removed_staves,CV_8U);
    for (int i = 0; i<number_labels; ++i)
    {
        int max_x=0;
        int min_x=-1;
        int max_y=0;
        int min_y=-1;
        for (int y = 0; y<removed_staves.rows; ++y)
        {
            for (int x = 0; x<removed_staves.cols; ++x)
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
        crop_area.width = 1+ max_x - min_x;
        crop_area.height = 1 + max_y - min_y;
        elements.push_back(removed_staves(crop_area));
    }
}

void Score::standardise_elements()
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
        std::string number_leading_zero = std::to_string(i);
        number_leading_zero.insert(number_leading_zero.begin(), 5 - number_leading_zero.length(), '0');
        std::string filename = "/home/jenny/Documents/Code/Coursework/OMR/Elements/element" + number_leading_zero + ".png";
        cv::imwrite(filename,standardised_elements[i-1]);
    }
}

std::vector<int> Score::get_staves()
{
    return staves;
}

cv::Mat Score::get_original_image()
{
    return original_image;
}

cv::Mat Score::get_binarized_image()
{
    return binarized_image;
}

cv::Mat Score::get_removed_staves()
{
    return removed_staves;
}

cv::Mat Score::get_connected_components()
{
    return coloured_connected_components;
}

std::vector<cv::Mat> Score::get_elements()
{
    return elements;
}

std::vector<cv::Mat> Score::get_standardised_elements()
{
    return standardised_elements;
}

bool Score::get_image_exists()
{
    return image_exists;
}

void Score::set_x_y(int x, int y)
{
    output_x = x;
    output_y = y;
}

void Score::set_filepath(QString input_filepath)
{
    filepath = input_filepath;
}

void Score::set_brightness_contrast(float input_contrast, int input_brightness)
{
    contrast = input_contrast;
    brightness = input_brightness;
}

void Score::proccess_image()
{
    if (image_exists == true)
    {
        load_mat();
        brightness_contrast();
        binarize();
        find_staves();
        remove_staves();
        find_connected_components();
    }
}

void Score::split_image()
{
    if (image_exists == true)
    {
        split_elements();
        standardise_elements();
    }
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
