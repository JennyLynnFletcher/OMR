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

    void on_remove_button_clicked();


private:
    Ui::MainWindow *ui;
    class Score
    {
    private:
        int output_x = 50;
        int output_y = 50;
        std::string filepath;
        std::vector<int> staves;
        cv::Mat original_image;
        cv::Mat binarized_image;
        cv::Mat removed_staves;
        cv::Mat coloured_connected_components;
        std::vector<cv::Mat> elements;
        std::vector<cv::Mat> standardised_elements;

        void load_mat();
        void brightness_contrast(cv::Mat image_to_edit);
        void binarize(cv::Mat new_image);
        int *histogram(cv::Mat image, int values[]);
        void find_staves(cv::Mat new_image);
        void remove_staves(cv::Mat image,std::vector<int> stave_values);
        void find_connected_components(cv::Mat image);
        void split_elements(cv::Mat image, cv::Mat label, int number_labels);
        void standardise_elements(std::vector<cv::Mat> elements, int x, int y);
    public:
        std::vector<int> get_staves();
        cv::Mat get_original_image();
        cv::Mat get_binarized_image();
        cv::Mat get_removed_staves();
        cv::Mat get_connected_components();
        std::vector<cv::Mat> get_elements();
        std::vector<cv::Mat> get_standardised_elements();

        void set_x_y(int x, int y);
        void set_filepath(std::string filepath);

    };

};

#endif // OMR_H
