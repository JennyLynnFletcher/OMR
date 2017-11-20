#ifndef SCORE_CLASS_H
#define SCORE_CLASS_H

class Score
{
private:
    int output_x;
    int output_y;
    float contrast;
    int brightness;
    QString filepath;
    bool image_exists;
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
    bool get_image_exists();

    void set_x_y(int x, int y);
    void set_filepath(QString input_filepath);
    void set_brightness_contrast(float input_contrast, int input_brightness);

    void proccess_image();
    void split_image();

};

#endif // SCORE_CLASS_H
