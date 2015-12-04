#define CPU_ONLY
#define USE_OPENCV
#include "caffe/caffe.hpp"
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using std::string;

class Classifier {
public:
    Classifier(const string& model_file,
               const string& trained_file);
    
    std::vector<float> Classify(const Mat& img, int N = 2);
    
private:
    std::vector<float> Predict(const Mat& img);
    
    void WrapInputLayer(std::vector<Mat>* input_channels);
    
    void Preprocess(const Mat& img,
                    std::vector<Mat>* input_channels);
    
private:
    shared_ptr<Net<float> > net_;
    Size input_geometry_;
    int num_channels_;
};



#endif  // USE_OPENCV
