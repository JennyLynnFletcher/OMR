#ifndef OMR_H
#define OMR_H

#include <vector>
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
    cv::Mat load_mat();
    cv::Mat brightness_contrast(cv::Mat image_to_edit);
    cv::Mat binarize(cv::Mat new_image);
    int *histogram(cv::Mat image, int values[]);
    std::vector<int> find_staves(cv::Mat new_image);
    cv::Mat remove_staves(cv::Mat image,std::vector<int> stave_values);
    std::vector<cv::Mat> find_connected_components(cv::Mat image);
    std::vector<cv::Mat> split_elements(cv::Mat image, cv::Mat label, int number_labels);
    std::vector<cv::Mat> standardise_elements(std::vector<cv::Mat> elements, int x, int y);
};

#endif // OMR_H
