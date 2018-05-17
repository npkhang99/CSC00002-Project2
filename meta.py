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
    # print(path)
    validate = {}
    test = {}
    train = {}
    uid = {}
    cnt = 0
    for folder in sorted(glob.glob(path + '*/')):
        print('Processing {:>30s}- uid: {}'.format(folder, cnt))
        name = folder.split('/')[-2]
        uid[name] = cnt
        cnt += 1
        files = glob.glob(folder + '*')
        random.shuffle(files)
        n = len(files)
        validate[name] = []
        test[name] = []
        train[name] = []
        for i in range(int(n * cfg.TEST_RATIO)):
            test[name].append(files[0])
            files.pop(0)
        for i in range(int(n * cfg.VALIDATE_RATIO)):
            validate[name].append(files[0])
            files.pop(0)
        for i in range(int(n * cfg.TRAIN_RATIO)):
            train[name].append(files[0])
            files.pop(0)
    
    # print meta files
    with open('validate.json', 'w') as f:
        f.write(json.dumps(validate, indent = 4))
        f.close()
    with open('test.json', 'w') as f:
        f.write(json.dumps(test, indent = 4))
        f.close()
    with open('train.json', 'w') as f:
        f.write(json.dumps(train, indent = 4))
        f.close()
    with open('uid.json', 'w') as f:
        f.write(json.dumps(uid, indent = 4))
        f.close()
    print('Meta files created')

def main(path):
    if path[-1] != '/':
        path += '/'
    create_meta_file(path)

if __name__ == '__main__':
    main(sys.argv[1])
