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

    pic_w, pic_h, pic_d = img.shape

    # img = scale_image(img, 1.5)

    # cv2.imshow('source', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (100, 100),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    face = np.array([], dtype = 'uint8')

    print(len(faces))

    if len(faces) > 0:
        print('Reading model...')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(cfg.MODEL_PATH)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 5), (255, 0, 0), 5)
        predict_id, percent = predict(recognizer, gray[y : y + h, x : x + w])
        name = get_name_with_uid(predict_id) + ' - {:.2f}'.format(percent)
        cv2.putText(img, name, (x, y + h), 0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite('image.jpg', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def show_video():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(cfg.MODEL_PATH)


    while True:
        ret, img = video.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (100, 100),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 5), (255, 255, 0), 2)
            predict_id, percent = predict(recognizer, gray[y : y + h, x : x + w])
            name = get_name_with_uid(predict_id) + ' - {:.2f}'.format(percent)
            cv2.putText(img, name, (x, y + h), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 1:
        show_video()
    else:
        print("ERROR")
