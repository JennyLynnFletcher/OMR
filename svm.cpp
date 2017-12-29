#include <iostream>
#include <vector>
#include <math.h>
#include <QPixmap>
#include <QtCore>
#include <iostream>
#include <cv.h>
#include <ml.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>


void CreateTrainTestHOG(std::vector<std::vector<float> > &trainHOG, std::vector<std::vector<float> > &testHOG, std::vector<cv::Mat> &trainCells, std::vector<cv::Mat> &testCells)
{
    cv::HOGDescriptor hog(
            cv::Size(20,20),cv::Size(10,10),cv::Size(5,5),cv::Size(10,10),9,1,-1,0,0.2,1,64,1);

    for(int y=0;y<int(trainCells.size());y++)
    {
        std::vector<float> descriptors;
        hog.compute(trainCells[y],descriptors);
        trainHOG.push_back(descriptors);
    }

    for(int y=0;y<int(testCells.size());y++)
    {
        std::vector<float> descriptors;
        hog.compute(testCells[y],descriptors);
        testHOG.push_back(descriptors);
    }
}


void ConvertVectortoMatrix(std::vector<std::vector<float> > &trainHOG, std::vector<std::vector<float> > &testHOG, cv::Mat &trainMat, cv::Mat &testMat)
{

    int descriptor_size = trainHOG[0].size();

    for(int i = 0;i<int(trainHOG.size());i++){
        for(int j = 0;j<descriptor_size;j++){
           trainMat.at<float>(i,j) = trainHOG[i][j];
        }
    }
    for(int i = 0;i<int(testHOG.size());i++){
        for(int j = 0;j<descriptor_size;j++){
            testMat.at<float>(i,j) = testHOG[i][j];
        }
    }
}

void getSVMParams(cv::ml::SVM *svm)
{
    std::cout << "Kernel type     : " << svm->getKernelType() << std::endl;
    std::cout << "Type            : " << svm->getType() << std::endl;
    std::cout << "C               : " << svm->getC() << std::endl;
    std::cout << "Degree          : " << svm->getDegree() << std::endl;
    std::cout << "Nu              : " << svm->getNu() << std::endl;
    std::cout << "Gamma           : " << svm->getGamma() << std::endl;
}

cv::Ptr<cv::ml::SVM> svmInit(float C, float gamma)
{
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(cv::ml::SVM::RBF);
  svm->setType(cv::ml::SVM::C_SVC);

  return svm;
}

void svmTrain(cv::Ptr<cv::ml::SVM> svm, cv::Mat &trainMat, std::vector<int> &trainLabels)
{
  cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainMat, cv::ml::ROW_SAMPLE, trainLabels);
  svm->train(td);
  svm->save("results/eyeGlassClassifierModel.yml");
}

void svmPredict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &testResponse, cv::Mat &testMat )
{
  svm->predict(testMat, testResponse);
}

void SVMevaluate(cv::Mat &testResponse, float &count, float &accuracy, std::vector<int> &testLabels)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    // cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
    if(testResponse.at<float>(i,0) == testLabels[i])
      count = count + 1;
  }
  accuracy = (count/testResponse.rows)*100;
}


