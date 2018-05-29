#!/usr/bin/env python3
import sys
import glob
import json
import cv2
import numpy as np

import dataset
import config as cfg

def train1():
    print("LBPHFaceRecognizer...")
    print("Preparing training files...", file = sys.stderr)
    faces, ids = dataset.load_train_images_and_ids(True,(300,300))
    print("Creating training model...", file = sys.stderr)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    print("Saving model...", file = sys.stderr)
    recognizer.save(cfg.MODEL_PATH)

def train2():
    print("FisherFaceRecognizer...")
    print("Preparing training files...", file = sys.stderr)
    faces, ids = dataset.load_train_images_and_ids(True,(300,300))
    print("Creating training model...", file = sys.stderr)
    recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    print("Saving model...", file = sys.stderr)
    recognizer.save(cfg.MODEL_PATH)
    
def train3():
    print("EigenFaceRecognizer...")
    print("Preparing training files...", file = sys.stderr)
    faces, ids = dataset.load_train_images_and_ids(True,(300,300))
    print("Creating training model...", file = sys.stderr)
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    print("Saving model...", file = sys.stderr)
    recognizer.save(cfg.MODEL_PATH)

def main(a):
    if int(a) == 1:
        train1()
    elif int(a) == 2:
        train2()
    elif int(a) == 3:
        train3()

if __name__ == '__main__':
    main(sys.argv[1])
