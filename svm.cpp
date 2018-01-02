#include "svm.h"

std::vector<cv::Mat> SVM::load_data(std::string path)
{
        DIR*    dir;
        dirent* pdir;
        std::vector<std::string> files;

        dir = opendir(path.c_str());

        while ((pdir = readdir(dir)))
        {
            files.push_back(pdir->d_name);
        }
        std::cout<<files[0]<<std::endl;

        std::vector<cv::Mat> images;
        for (int i = 0; i < int(files.size());i++)
        {
           images.push_back(cv::imread(files[i]));
        }

        return images;
}

std::vector<std::vector<float>> SVM::CreateTrainTestHOG(std::vector<cv::Mat> &cells)
{
    cv::HOGDescriptor hog(
            cv::Size(50,50),cv::Size(20,20),cv::Size(10,10),cv::Size(20,20),9,1,-1,0,0.2,0,64,1);
    std::vector<std::vector<float>> HOG;
    for(int y=0;y<int(cells.size());y++)
    {
        std::vector<float> descriptors;
        hog.compute(cells[y],descriptors);
        HOG.push_back(descriptors);
    }
    return HOG;
}


cv::Mat SVM::ConvertVectortoMatrix(std::vector<std::vector<float> > &HOG, cv::Mat &Mat)
{

    int descriptor_size = HOG[0].size();

    for(int i = 0;i<int(HOG.size());i++){
        for(int j = 0;j<descriptor_size;j++){
           Mat.at<float>(i,j) = HOG[i][j];
        }
    }
    return Mat;
}

void SVM::getSVMParams(cv::ml::SVM *svm)
{
    std::cout << "Kernel type     : " << svm->getKernelType() << std::endl;
    std::cout << "Type            : " << svm->getType() << std::endl;
    std::cout << "C               : " << svm->getC() << std::endl;
    std::cout << "Degree          : " << svm->getDegree() << std::endl;
    std::cout << "Nu              : " << svm->getNu() << std::endl;
    std::cout << "Gamma           : " << svm->getGamma() << std::endl;
}

cv::Ptr<cv::ml::SVM> SVM::svmInit(float C, float gamma)
{
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(cv::ml::SVM::RBF);
  svm->setType(cv::ml::SVM::C_SVC);

  return svm;
}

void SVM::svmTrain(cv::Ptr<cv::ml::SVM> svm, cv::Mat &trainMat, std::vector<int> &trainLabels)
{
  cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainMat, cv::ml::ROW_SAMPLE, trainLabels);
  svm->train(td);
  svm->save("results/OMR_SVM_Results.yml");
}

void SVM::svmPredict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &testResponse, cv::Mat &testMat )
{
  svm->predict(testMat, testResponse);
}

void SVM::SVMevaluate(cv::Mat &testResponse, float &count, float &accuracy, std::vector<int> &testLabels)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    if(testResponse.at<float>(i,0) == testLabels[i])
      count = count + 1;
  }
  accuracy = (count/testResponse.rows)*100;
}

void SVM::train_SVM()
{
    std::vector<cv::Mat> trainCells = load_data("/home/jenny/Documents/Code/Coursework/OMR/Train_Data/");
    std::vector<int> trainLabels;

    std::vector<std::vector<float>> trainHOG = CreateTrainTestHOG(trainCells);

    cv::Mat trainMat = ConvertVectortoMatrix(trainHOG,trainMat);

    float C = 12.5, gamma = 0.5;

    cv::Mat testResponse;
    cv::Ptr<cv::ml::SVM> model = svmInit(C, gamma);

    svmTrain(model, trainMat, trainLabels);
}

void SVM::classify_SVM()
{
    cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::load("results/OMR_SVM_Results.yml");
    std::vector<cv::Mat> testCells = load_data("/home/jenny/Documents/Code/Coursework/OMR/Elements/");
    std::vector<std::vector<float>> testHOG = CreateTrainTestHOG(testCells);
    cv::Mat testMat = ConvertVectortoMatrix(testHOG,testMat);
    cv::Mat testResponse;
    svmPredict(model, testResponse, testMat);
}

void SVM::run_SVM()
{
    std::vector<cv::Mat> trainCells = load_data("/home/jenny/Documents/Code/Coursework/OMR/Train_Data/");
    std::vector<cv::Mat> testCells = load_data("/home/jenny/Documents/Code/Coursework/OMR/Elements/");
    std::vector<int> trainLabels;

    std::vector<std::vector<float>> trainHOG = CreateTrainTestHOG(trainCells);
    std::vector<std::vector<float>> testHOG = CreateTrainTestHOG(testCells);

    cv::Mat trainMat = ConvertVectortoMatrix(trainHOG,trainMat);
    cv::Mat testMat = ConvertVectortoMatrix(testHOG,testMat);

    float C = 12.5, gamma = 0.5;

    cv::Mat testResponse;
    cv::Ptr<cv::ml::SVM> model = svmInit(C, gamma);

    svmTrain(model, trainMat, trainLabels);

    svmPredict(model, testResponse, testMat);
}
