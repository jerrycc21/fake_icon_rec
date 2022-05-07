import os
import argparse
import numpy as np

def cal_acc(pred_file, target_file):
    preds = []
    gts = []
    for line in open(pred_file):
        line = line.strip().split()
        preds.append(line[1])
    for line in open(target_file):
        line = line.strip().split()
        gts.append(line[1])
    correct = 0
    for i in range(len(preds)):
        if preds[i] == gts[i]:
            correct += 1
    return correct / len(preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gt', default='./data/test_files.txt', type=str,
            help='groundtruth')
    parser.add_argument('--pred', default='./test_results.txt', type=str,
            help='test results')
    args = parser.parse_args()
    acc = cal_acc(args.pred, args.gt)
    print("accuracy is: %f" % acc)

