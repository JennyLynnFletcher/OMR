#include "svm.h"


std::vector<int> SVM::load_labels(std::string path)
{
    std::vector<int> labels;
    std::ifstream label_file(path);
    std::string line;
    while (std::getline(label_file, line))
        {
            labels.push_back(std::stoi(line));
        }
    return labels;
}


std::vector<cv::Mat> SVM::load_images(std::string path)
{
        DIR*    dir;
        dirent* pdir;
        std::vector<std::string> files;

        dir = opendir(path.c_str());

        while ((pdir = readdir(dir)))
        {
            files.push_back(pdir->d_name);
        }
        std::sort(files.begin(),files.end());
        std::vector<cv::Mat> images;
        for (int i = 0; i < int(files.size());i++)
        {
           if (cv::imread(path + files[i]).empty() == false)
           {
            images.push_back(cv::imread(path + files[i]));
            std::cout<<files[i]<<std::endl;
           }
        }

        return images;
}

std::vector<std::vector<float>> SVM::create_train_test_HOG(std::vector<cv::Mat> &cells)
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


cv::Mat SVM::convert_vector_to_matrix(std::vector<std::vector<float>> &HOG)
{

    int descriptor_size = HOG[0].size();
    cv::Mat mat(HOG.size(),descriptor_size,CV_32FC1);

    for(int i = 0;i<int(HOG.size());i++){
        for(int j = 0;j<descriptor_size;j++){
           mat.at<float>(i,j) = HOG[i][j];
        }
    }
    return mat;
}

void SVM::get_SVM_params(cv::ml::SVM *svm)
{
    std::cout << "Kernel type     : " << svm->getKernelType() << std::endl;
    std::cout << "Type            : " << svm->getType() << std::endl;
    std::cout << "C               : " << svm->getC() << std::endl;
    std::cout << "Degree          : " << svm->getDegree() << std::endl;
    std::cout << "Nu              : " << svm->getNu() << std::endl;
    std::cout << "Gamma           : " << svm->getGamma() << std::endl;
}

cv::Ptr<cv::ml::SVM> SVM::svm_init(float C, float gamma)
{
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(cv::ml::SVM::RBF);
  svm->setType(cv::ml::SVM::C_SVC);

  return svm;
}

void SVM::svm_train(cv::Ptr<cv::ml::SVM> svm, cv::Mat &train_mat, std::vector<int> &train_labels)
{
  cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(train_mat, cv::ml::ROW_SAMPLE, train_labels);
  svm->train(td);
  svm->save("/home/jenny/Documents/Code/Coursework/OMR/results/OMR_SVM_Results.yml");
}

void SVM::svm_predict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &test_response, cv::Mat &test_mat )
{
  svm->predict(test_mat, test_response);
  std::ofstream results_output;
  results_output.open("/home/jenny/Documents/Code/Coursework/OMR/results/results.txt");
  for(int i = 0; i < test_response.rows; i++)
  {
     results_output<<(std::to_string(test_response.at<float>(i,0)) + "\n");
  }
  results_output.close();
}

void SVM::SVM_evaluate(cv::Mat &test_response, float &count, float &accuracy, std::vector<int> &test_labels)
{
  for(int i = 0; i < test_response.rows; i++)
  {
    if(test_response.at<float>(i,0) == test_labels[i])
      count = count + 1;
  }
  accuracy = (count/test_response.rows)*100;
}

void SVM::train_SVM()
{
    std::vector<cv::Mat> train_cells = load_images("/home/jenny/Documents/Code/Coursework/OMR/Train_Data/");
    std::vector<int> train_labels = load_labels("/home/jenny/Documents/Code/Coursework/OMR/train_values");

    std::vector<std::vector<float>> train_HOG = create_train_test_HOG(train_cells);

    cv::Mat train_mat = convert_vector_to_matrix(train_HOG);

    float C = 12.5, gamma = 0.5;

    cv::Mat test_response;
    cv::Ptr<cv::ml::SVM> model = svm_init(C, gamma);

    svm_train(model, train_mat, train_labels);
}

void SVM::classify_SVM()
{
    cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::load("/home/jenny/Documents/Code/Coursework/OMR/results/OMR_SVM_Results.yml");
    std::vector<cv::Mat> test_cells = load_images("/home/jenny/Documents/Code/Coursework/OMR/Elements/");
    std::vector<std::vector<float>> test_HOG = create_train_test_HOG(test_cells);
    cv::Mat test_mat = convert_vector_to_matrix(test_HOG);
    cv::Mat test_response;
    svm_predict(model, test_response, test_mat);
}

void SVM::run_SVM()
{
    std::vector<cv::Mat> train_cells = load_images("/home/jenny/Documents/Code/Coursework/OMR/Train_Data/");
    std::vector<int> train_labels = load_labels("/home/jenny/Documents/Code/Coursework/OMR/train_values");
    std::vector<cv::Mat> test_cells = load_images("/home/jenny/Documents/Code/Coursework/OMR/Elements/");

    std::vector<std::vector<float>> train_HOG = create_train_test_HOG(train_cells);
    std::vector<std::vector<float>> test_HOG = create_train_test_HOG(test_cells);

    cv::Mat train_mat = convert_vector_to_matrix(train_HOG);
    cv::Mat test_mat = convert_vector_to_matrix(test_HOG);

    float C = 12.5, gamma = 0.5;

    cv::Mat test_response;
    cv::Ptr<cv::ml::SVM> model = svm_init(C, gamma);

    svm_train(model, train_mat, train_labels);

    svm_predict(model, test_response, test_mat);
}
