//
//  Detector.hpp
//  face_detector
//
//  Created by Yukai Wang on 04/12/15.
//  Copyright © 2015 Yukai Wang. All rights reserved.
//

#ifndef Detector_hpp
#define Detector_hpp

#include <fstream>
#include "Classifier.hpp"


const int TAILLE_IMG = 36;
const float REDUCTION_RATE = 1.2;
const string PYTHON_CMD = "python dbscan.py";

struct Tag {
    int realX;
    int realY;
    int realWidth;
    int realHeight;
};

class Detector
{
public:
    Detector(const string& model_file,
                  const string& trained_file,
                  const Mat& img);
    ~Detector();
    void Detect();
    void Clustering();
    void Draw();
    
protected:
    void FrameIterator(Mat actualImg);
    void WriteCSV();
    void ReadCSV();
    void Transform();
    
    Classifier* classifier;
    Mat originalImg;
    int nbIterator;
    std::vector<Tag> tags;
    std::map<int, std::vector<int> > clusters;
    std::vector<Tag> results;
};


#endif /* Detector_hpp */
