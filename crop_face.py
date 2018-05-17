#!/usr/bin/env python3
import numpy
import cv2
import sys
import time
import os
import glob
import logging

# asdfasdfasdf

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level = logging.INFO, format = FORMAT)
log = logging.getLogger('Log')

def crop_faces(path, output_dir):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if path[-1] != '/':
        path += '/'
    
    output_dir = output_dir + path.split('/')[-2] + '/'

    if os.path.exists(path):
        for file_name in sorted(glob.glob(path + '*.jpg')):
            log.info("Processing file {}...".format(file_name))

            # read image
            img = cv2.imread(file_name)

            # convert to gray scale and detect faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (200, 200),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            # init cropped face array
            face = numpy.array([], dtype = 'uint8')

            # is it empty??
            if faces != []:
                log.info("Found {} face(s)".format(len(faces)))
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

            file_name = output_dir + file_name.split('/')[-1]
            # for each face found
            for k in range(len(faces)):
                # get the face's coords and dimensions
                x, y, h, w = faces[k]

                # resize cropped face array to the correct size
                face.resize((h, w, 3))

                # copy to the face to face array
                for i in range(w):
                    for j in range(h):
                        face[i][j] = img[y + i][x + j]

                # write face to file
                if len(faces) != 1:
                    cv2.imwrite(os.path.splitext(file_name)[0] + '_' + str(k) + os.path.splitext(file_name)[1], face)
                else:
                    cv2.imwrite(file_name, face)

            # cv2.imshow('source', img)
            # cv2.imshow('cropped', face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

if __name__ == '__main__':
    if sys.argv[1][-1] != '/':
        sys.argv[1] += '/'
    if not os.path.exists(sys.argv[1] + '../faces'):
        os.mkdir(sys.argv[1] + '../faces')
    for folder in glob.glob(sys.argv[1] + '*/'):
        crop_faces(folder, sys.argv[1] + '../faces/')
