//
//  Detector.cpp
//  face_detector
//
//  Created by Yukai Wang on 04/12/15.
//  Copyright © 2015 Yukai Wang. All rights reserved.
//

#include "Detector.hpp"

Detector::Detector(const string& model_file,
                             const string& trained_file,
                             const Mat& img) {
    classifier = new Classifier(model_file, trained_file);
    originalImg = img;
    nbIterator = 0;
}

Detector::~Detector(){
    delete classifier;
}

void Detector::FrameIterator(Mat actualImg ) {
    int actualWidth = actualImg.size().width;
    int actualHeight = actualImg.size().height;
    
    for (int i = 0; i <= actualHeight-TAILLE_IMG; i += 1) {
        for (int j = 0; j <= actualWidth-TAILLE_IMG; j += 1) {
            Mat smallFrame = actualImg(Rect(j, i, TAILLE_IMG, TAILLE_IMG));
            std::vector<float> predictions = classifier->Classify(smallFrame);
            if (predictions[0] < 0.4) {
                Tag tag;
                tag.realX = pow(REDUCTION_RATE, nbIterator) * i;
                tag.realY = pow(REDUCTION_RATE, nbIterator) * j;
                tag.realWidth = pow(REDUCTION_RATE, nbIterator) * TAILLE_IMG;
                tag.realHeight = pow(REDUCTION_RATE, nbIterator) * TAILLE_IMG;
                tags.push_back(tag);
            }
        }
    }
}

void Detector::Draw() {
    if (results.size() == 0) {
        std::cout << "No result" << std::endl;
    }
    for (size_t i = 0; i < results.size(); ++i) {
        Tag tag = results[i];
        std::cout << "realX: " << tag.realX << std::endl;
        std::cout << "realY: " << tag.realY << std::endl;
        std::cout << "realWidth: " << tag.realWidth << std::endl;
        std::cout << "realHeight: " << tag.realHeight << std::endl;
        rectangle(originalImg, Point(tag.realX, tag.realY), Point(tag.realX+tag.realWidth, tag.realY+tag.realHeight), Scalar(255, 0, 0));
    }
}

void Detector::Detect() {
    Mat actualImg = originalImg.clone();
    int actualWidth = actualImg.size().width;
    int actualHeight = actualImg.size().height;
    while (actualWidth >= TAILLE_IMG && actualHeight >= TAILLE_IMG) {
        FrameIterator(actualImg);
        Size size(actualWidth/REDUCTION_RATE,actualHeight/REDUCTION_RATE);
        resize(actualImg,actualImg,size);
        actualWidth = actualImg.size().width;
        actualHeight = actualImg.size().height;
        nbIterator++;
    }
}

void Detector::WriteCSV() {
    std::ofstream output("MeanshiftInput.csv");
    for (int i = 0; i < tags.size(); i++) {
        int centerX = tags[i].realX + tags[i].realWidth/2;
        int centerY = tags[i].realY + tags[i].realHeight/2;
        output << centerX << ";" << centerY << std::endl;
    }
}

void Detector::ReadCSV() {
    std::ifstream input("MeanshiftOutput.csv");
    if(!input.is_open()){
        std::cerr << "failed to open file.";
        return;
    }
    const int MAXSIZE = 20;
    char oneLine[MAXSIZE];
    int i = 0;
    while(input.getline(oneLine,MAXSIZE)) {
        string str(oneLine);
        string str_cluster = str.substr(0, str.find_first_not_of(';'));
        int nbCluster = stoi(str);
        clusters[nbCluster].push_back(i++);
    }
}

void Detector::Transform() {
    std::map<int, std::vector<int> >::iterator it;
    for (it = clusters.begin(); it != clusters.end(); ++it) {
        int sumX = 0;
        int sumY = 0;
        int sumWidth = 0;
        int sumHeight = 0;
        int size = (int)it->second.size();
        vector<int> numbers = it->second;
        for (int n = 0; n < size; n++) {
            Tag tag = tags[numbers[n] ];
            sumX += tag.realX;
            sumY += tag.realY;
            sumWidth += tag.realWidth;
            sumHeight += tag.realHeight;
        }
        Tag averageTag = {sumX/size, sumY/size, sumWidth/size, sumHeight/size};
        results.push_back(averageTag);
    }
}

void Detector::Clustering() {
    if (tags.size() == 0) {
        return;
    }
    WriteCSV();
    system(PYTHON_CMD.c_str());
    ReadCSV();
    Transform();
}
