#ifndef SVM_H
#define SVM_H

#include <iostream>
#include <vector>
#include <math.h>
#include <QPixmap>
#include <opencv2/opencv.hpp>
#include <QtCore>
#include <iostream>
#include <dirent.h>
#include <cv.h>
#include <ml.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>


class SVM
{
private:
    std::vector<int> load_labels(std::string path);
    std::vector<cv::Mat> load_images(std::string path);
    std::vector<std::vector<float>>create_train_test_HOG(std::vector<cv::Mat> &cells);
    cv::Mat convert_vector_to_matrix(std::vector<std::vector<float> > &HOG);
    void get_SVM_params(cv::ml::SVM *svm);
    cv::Ptr<cv::ml::SVM> svm_init(float C, float gamma);
    void svm_train(cv::Ptr<cv::ml::SVM> svm, cv::Mat &train_mat, std::vector<int> &train_labels);
    std::vector<int> svm_predict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &test_response, cv::Mat &test_mat );
    void SVM_evaluate(cv::Mat &test_response, float &count, float &accuracy, std::vector<int> &test_labels);
public:
    void run_SVM();
    std::vector<int> classify_SVM();
    void train_SVM();
};

#endif // SVM_H
