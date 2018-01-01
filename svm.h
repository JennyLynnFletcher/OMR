#ifndef SVM_H
#define SVM_H

#include <cv.h>
#include <ml.h>

class SVM
{
private:
    std::vector<cv::Mat> load_data(std::string path);
    std::vector<std::vector<float>>CreateTrainTestHOG(std::vector<cv::Mat> &cells);
    cv::Mat ConvertVectortoMatrix(std::vector<std::vector<float> > &HOG, cv::Mat &Mat);
    void getSVMParams(cv::ml::SVM *svm);
    cv::Ptr<cv::ml::SVM> svmInit(float C, float gamma);
    void svmTrain(cv::Ptr<cv::ml::SVM> svm, cv::Mat &trainMat, std::vector<int> &trainLabels);
    void svmPredict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &testResponse, cv::Mat &testMat );
    void SVMevaluate(cv::Mat &testResponse, float &count, float &accuracy, std::vector<int> &testLabels);
public:
    void run_SVM();
};

#endif // SVM_H
