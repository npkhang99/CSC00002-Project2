#!/usr/bin/env python3
import sys
import glob
import json
import random
import json

import config as cfg

def create_meta_file(path):
    if path[-1] != '/':
        path += '/'
    pics = glob.glob(path + '*.jpg')
    # print(path)
    n = len(pics)
    random.shuffle(pics)
    meta = {
        "TRAINING": [],
        "VALIDATING": [],
        "TESTING": []
    }
    with open(path + 'meta.json', 'w') as f:
        for i in range(int(n * cfg.TRAIN_RATIO)):
            # f.write(pics[0] + '\n')
            meta["TRAINING"].append(pics[0])
            pics.pop(0)
        for i in range(int(n * cfg.VALIDATE_RATIO)):
            # f.write(pics[0] + '\n')
            meta["VALIDATING"].append(pics[0])
            pics.pop(0)
        for i in range(int(n * cfg.TEST_RATIO)):
            # f.write(pics[0] + '\n')
            meta["TESTING"].append(pics[0])
            pics.pop(0)
        # print(meta)
        f.write(json.dumps(meta, indent = 4))
        f.close()
    print('Meta file created for {}'.format(path))

def main(path):
    if path[-1] != '/':
        path += '/'
    for folder in glob.glob(path + '*/'):
        create_meta_file(folder)

if __name__ == '__main__':
    main(sys.argv[1])
