import numpy as np
import cv2
import os
import sys
import json

def convert_image_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def resize_image(img, size):
    img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
    return img

def scale_image(img, scale):
    h, w, d = img.shape
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation = cv2.INTER_AREA)
    return img

def get_name_with_uid(uid):
    uid_json = json.load(open("uid.json",'r'))
    name = 'Unknow'
    for user in uid_json:
        if uid_json[user] == uid:
            name = user
            break
    return name

def load_train_images_and_ids(resize = False, size = (200,200)):
    faces = []
    ids = []
    train = json.load(open("train.json",'r'))
    uid = json.load(open("uid.json",'r'))
    for name in train:
        for face in train[name]:
            ids.append(uid[name])
            img = convert_image_to_grayscale(cv2.imread(face))
            if resize:
                img = resize_image(img, size)
            faces.append(img)
    return faces, ids

def load_test_images_and_ids(resize = False, size = (200,200)):
    faces = []
    ids = []
    train = json.load(open("test.json",'r'))
    uid = json.load(open("uid.json",'r'))
    for name in train:
        for face in train[name]:
            ids.append(uid[name])
            img = convert_image_to_grayscale(cv2.imread(face))
            if resize:
                img = resize_image(img, size)
            faces.append(img)
    return faces, ids
