#!/usr/bin/env python3
import sys
import glob
import json
import cv2
import numpy as np

from dataset import resize_image, scale_image, get_name_with_uid
from eval import predict
import config as cfg

def main(picture):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(picture)

    # img = scale_image(img, 1.5)

    # cv2.imshow('source', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (10, 10),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    face = np.array([], dtype = 'uint8')

    print(len(faces))

    if len(faces) > 0:
        print('Reading model...')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(cfg.MODEL_PATH)

    for k in range(len(faces)):
        # get the face's coords and dimensions
        x, y, h, w = faces[k]

        # resize cropped face array to the correct size
        face.resize((h, w, 3))

        # copy to the face to face array
        for i in range(w):
            for j in range(h):
                face[i][j] = img[y + i][x + j]

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        predict_id = predict(recognizer, face_gray)
        # show face and id
        cv2.imshow(str(get_name_with_uid(predict_id)), face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("ERROR")
