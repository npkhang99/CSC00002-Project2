#!/usr/bin/env python3
import sys
import glob
import json
import cv2
import numpy as np

import dataset
import config as cfg

def main():
    print("Preparing validating files...", file = sys.stderr)
    faces, ids = dataset.load_test_images_and_ids(True, (300,300))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(cfg.MODEL_PATH)
    print("Evaluating...", file = sys.stderr)
    n = len(faces)
    correct = 0
    for i in range(n):
        Id, percent = recognizer.predict(faces[i])
        print('Face {:3d} as {:2d}, recognized as {:2d} with confidence of {:6.2f} - {}'.format(i, ids[i], Id, percent, "correct" if ids[i] == Id else "fail"))
        if ids[i] == Id:
            correct += 1
    print("Correct percentage: {}/{}, {:.2f} percent".format(correct, n, correct / n * 100))

if __name__ == '__main__':
    main()