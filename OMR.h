#ifndef OMR_H
#define OMR_H

#include <vector>
#include <string>
#include <QWidget>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <QPixmap>
#include <QtCore>
#include <cv.h>


#include <QMainWindow>
#include <score_class.h>

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

    void on_svm_train_clicked();

private:
    Ui::MainWindow *ui;
    bool original_image_selected = true;
    bool binarized_image_selected = false;
    bool removed_staves_selected = false;
    bool connected_components_selected = false;
    float contrast_slider_value = -1;
    int brightness_slider_value = -1;
    Score score_to_read;
};

#endif // OMR_H
