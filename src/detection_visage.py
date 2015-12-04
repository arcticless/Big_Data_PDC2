#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'NHS'
import caffe
import numpy as np
import os
src_dir = os.path.dirname(os.path.realpath(__file__))
import cv2
import argparse
from blist import sortedlist

class DetecteurVisage:
    def __init__(self):
        self.net = caffe.Net("facenet.prototxt","facenet.caffemodel",caffe.TEST)
        self.WINDOW_SIZE = (36,36) #Size of filter caffe
        self.STRIDE = 2 # intervals at which to apply the filters to the input (2 pixel)
        self.SCALE_FACTOR = 1.2 # image scale factor
        self.THRESHOLD = 0.75 # probability to conclure if found a face at a position. Here, we use "Softmax" function for the output layer "prob" (final layer in parameter file "facenet.prototxt")

    def execute(self,images_test_dir):
        # Delete all images in the result folder
        for img_file in os.listdir(src_dir+"/result/"):
            file_path = os.path.join(src_dir+"/result/", img_file)
            try:
                if os.path.splitext(img_file)[1][1:] == "pgm" or os.path.splitext(img_file)[1][1:] == "jpg" or os.path.splitext(img_file)[1][1:] == "jpeg" or os.path.splitext(img_file)[1][1:] == "png":
                    os.unlink(file_path)
            except Exception, e:
                print e
        cv2.namedWindow('detection_visage', cv2.WINDOW_AUTOSIZE)
        for img_file in os.listdir(images_test_dir):
            # Check image files in test folder
            if os.path.splitext(img_file)[1][1:] == "pgm" or os.path.splitext(img_file)[1][1:] == "jpg" or os.path.splitext(img_file)[1][1:] == "jpeg" or os.path.splitext(img_file)[1][1:] == "png":
                print "Fichier: "+ img_file
                img_file_path = images_test_dir+"/"+img_file
                # Read the input image
                img_cv = cv2.imread(img_file_path)
                img_height, img_width = img_cv.shape[:2]
                found = False
                count = 0
                all_boxes = []
                # Convert BGR image to gray
                img_cv_gray = cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
                for py_img in pyramid(img_cv_gray,scale = self.SCALE_FACTOR):
                    scale = pow(self.SCALE_FACTOR,count)
                    count += 1
                    #loop over the sliding window for each layer of the pyramid
                    for (x, y, window) in sliding_window(py_img, self.STRIDE, self.WINDOW_SIZE):
                        # Apply the CNN to position (x,y)
                        window_input = window[np.newaxis,np.newaxis,:]
                        self.net.blobs['data'].reshape(*window_input.shape)
                        self.net.blobs['data'].data[...] = window_input
                        out = self.net.forward()
                        # get output value of the final layer 'prob'
                        prob_val = out['prob'][0][1]
                        # Check output value (probability) is greater than the given threshold (a face appears
                        # probably at this position (x,y))
                        if prob_val >= self.THRESHOLD:
                            box = generateBoundingBox(x, y, prob_val, self.WINDOW_SIZE, scale,img_width,img_height)
                            all_boxes.append(box)
                if all_boxes:
                    found = True
                    all_boxes = removeNeighborsBox(all_boxes)
                #draw all rectangles at faces positions
                for box in all_boxes:
                    cv2.rectangle(img_cv,(box[0], box[1]),(box[2],box[3]),(0,0,255))
                    cv2.imshow('detection_visage',img_cv)
                if found == True:
                    #Pause the program to see the result and press the enter key to continue the program
                    raw_input("Presser 'enter' pour continuer...")
                    cv2.imwrite(src_dir+"/result/"+img_file,img_cv)
        # Pause the program to see the last result and press the enter key to terminate the program
        raw_input("Presser 'enter' pour terminer...")
        cv2.destroyAllWindows()

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, len(image)-windowSize[1], stepSize):
        for x in xrange(0, len(image[0])-windowSize[0], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, scale, minSize=(70, 70)):
    # return the original image
    yield np.array(image)
    # keep looping over the pyramid
    while True:
        # resize the image
        new_width = int(image.shape[1]/scale)
        new_height = int(image.shape[0]/scale)
        image = cv2.resize(image,(new_width, new_height))
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if new_width < minSize[0] or new_height < minSize[1]:
            break
        # return the next image in the pyramid
        yield np.array(image)

# Rectangle
def generateBoundingBox(x,y,prob_val,window_size, scale,img_width,img_height):
    real_x = int(x * scale)
    real_y = int(y * scale)
    real_w = int(window_size[0] * scale)
    real_h = int(window_size[1] * scale)
    print 'Visage détecté à Rect('+str(real_x)+','+str(real_y)+','+str(real_x+real_w)+','+str(real_y+real_h)+') prob='+str(prob_val)
    return [real_x, real_y, min(real_x+real_w,img_width), min(real_y+real_h,img_height)]

# Remove neighbors rectangle
def removeNeighborsBox(boxes):
    len_boxes = len(boxes)
    i = 0
    #sort the rectangles list by its coordinates vertex points
    boxes = sortedlist(boxes,key=lambda x:(x[0],x[1],x[2],x[3]))
    while i < len_boxes-1:
        box1 = boxes[i]
        j = i+1
        while j < len_boxes:
            box2 = boxes[j]
            if box2[0] >= box1[0] and box2[1] >= box1[1] and box2[2]<=box1[2] and box2[3]<=box1[3]:
                del boxes[j]
                len_boxes -= 1
                continue
            elif box2[0] <= box1[0] and box2[1] <= box1[1] and box2[2] >= box1[2] and box2[3] >= box1[3]:
                del boxes[i]
                len_boxes -= 1
                i -= 1
                break
            j += 1
        i += 1
    i = 0
    while i < len_boxes-1:
        box1 = boxes[i]
        j = i+1
        while j < len_boxes:
            box2 = boxes[j]
            # If the distances between 2 vertex points same x-coordinate or y-coordinate of 2 rectangles is lower than 10
            if (box2[0]-box1[0] < 10 and box2[1]==box1[1]) or (box2[1]-box1[1] < 10 and box2[0]==box1[0]):
                del boxes[i]
                i -= 1
                del boxes[j-1]
                len_boxes -= 1
                boxes.add([min(box1[0],box2[0]),min(box1[1],box2[1]),max(box1[2],box2[2]),max(box1[3],box2[3])])
                break
            j += 1
        i += 1
    return boxes

#Find max, min between 2 values
def min(x,y):
    if x>y:
        return y
    return x

def max(x,y):
    if x>y:
        return x
    return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Détection de visage')
    parser.add_argument('repertoire', metavar='repertoire', nargs='?', help='Chemin du répertoire d\'images',default=src_dir+"/images")
    args = parser.parse_args()
    dectect_visage = DetecteurVisage()
    dectect_visage.execute(args.repertoire)
