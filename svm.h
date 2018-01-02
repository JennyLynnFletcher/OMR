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
    std::vector<std::vector<float>>CreateTrainTestHOG(std::vector<cv::Mat> &cells);
    cv::Mat ConvertVectortoMatrix(std::vector<std::vector<float> > &HOG);
    void getSVMParams(cv::ml::SVM *svm);
    cv::Ptr<cv::ml::SVM> svmInit(float C, float gamma);
    void svmTrain(cv::Ptr<cv::ml::SVM> svm, cv::Mat &trainMat, std::vector<int> &trainLabels);
    void svmPredict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &testResponse, cv::Mat &testMat );
    void SVMevaluate(cv::Mat &testResponse, float &count, float &accuracy, std::vector<int> &testLabels);
public:
    void run_SVM();
    void classify_SVM();
    void train_SVM();
};

#endif // SVM_H
