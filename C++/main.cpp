//
//  main.cpp
//  face_detector
//
//  Created by Yukai Wang on 01/12/15.
//  Copyright © 2015 Yukai Wang. All rights reserved.
//

#include "Detector.hpp"

#ifdef USE_OPENCV
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
        << " deploy.prototxt network.caffemodel img.jpg" << std::endl;
        return 1;
    }
    
    ::google::InitGoogleLogging(argv[0]);
    
    string model_file   = argv[1];
    string trained_file = argv[2];
    string file = argv[3];
//    Classifier classifier(model_file, trained_file);
    
    std::cout << "---------- Prediction for "
    << file << " ----------" << std::endl;
    
    Mat img = imread(file);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    
    Detector detector(model_file, trained_file, img);
    detector.Detect();
    detector.Clustering();
    detector.Draw();
//    std::vector<float> predictions = classifier.Classify(img);
    
    /* Print the top N predictions. */
//    for (size_t i = 0; i < predictions.size(); ++i) {
//        float p = predictions[i];
//        std::cout << std::fixed << std::setprecision(4) << p << std::endl;
//    }
//
    imwrite( "output.jpg", img );
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", img );                // Show our image inside it.
    waitKey(0);
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
