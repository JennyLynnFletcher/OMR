#ifndef OMR_H
#define OMR_H

#include <vector>
#include <string>
#include <QWidget>
#include <opencv2/opencv.hpp>

#include <QMainWindow>

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

    void on_enter_button_clicked();

    void on_update_image_clicked();

    void on_go_button_clicked();

    void on_original_image_clicked();

    void on_binarized_image_clicked();

    void on_removed_staves_clicked();

    void on_connected_components_clicked();

private:
    Ui::MainWindow *ui;
    bool original_image_selected = true;
    bool binarized_image_selected = false;
    bool removed_staves_selected = false;
    bool connected_components_selected = false;
    float contrast_slider_value = -1;
    int brightness_slider_value = -1;

    class Score
    {
    private:
        int output_x;
        int output_y;
        float contrast;
        int brightness;
        QString filepath;
        int number_labels;
        std::vector<int> staves;
        cv::Mat original_image;
        cv::Mat BC_image;
        cv::Mat binarized_image;
        cv::Mat removed_staves;
        cv::Mat label_image;
        cv::Mat coloured_connected_components;
        std::vector<cv::Mat> elements;
        std::vector<cv::Mat> standardised_elements;


        void load_mat();
        void brightness_contrast();
        void binarize();
        int *histogram(int values[]);
        void find_staves();
        void remove_staves();
        void find_connected_components();
        void split_elements();
        void standardise_elements();
    public:
        std::vector<int> get_staves();
        cv::Mat get_original_image();
        cv::Mat get_binarized_image();
        cv::Mat get_removed_staves();
        cv::Mat get_connected_components();
        std::vector<cv::Mat> get_elements();
        std::vector<cv::Mat> get_standardised_elements();

        void set_x_y(int x, int y);
        void set_filepath(QString input_filepath);
        void set_brightness_contrast(float input_contrast, int input_brightness);

        void proccess_image();

    };
    Score score_to_read;
};

#endif // OMR_H
