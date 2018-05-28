#!/usr/bin/env python3
import sys
import glob
import json
import cv2
import numpy as np

import dataset
import config as cfg

def train():
    print("Preparing training files...", file = sys.stderr)
    faces, ids = dataset.load_train_images_and_ids()
    print("Creating training model...", file = sys.stderr)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    print("Saving model...", file = sys.stder)
    recognizer.save(cfg.MODEL_PATH)

def main():
    train()

if __name__ == '__main__':
    main()
