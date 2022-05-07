import os
import argparse
import time
import torch
import torch.nn as nn
import imageio
import numpy as np
import models
import cv2
import cairosvg
from tqdm import tqdm
from PIL import Image
from IPython import embed

def get_img(image_name):
    root_dir = "./data/icons"
    if image_name.split(".")[1] in ['png', 'jpg']:
        img = cv2.imread(os.path.join(root_dir, image_name))
    elif image_name.split(".")[1] == 'gif':
        t = imageio.mimread(os.path.join(root_dir, image_name), memtest=False)
        if len(t[0].shape) < 3:
            img = t[0][:, :]
        else:
            img = t[0][:, :, :3]
    elif image_name.split(".")[1] == 'svg':
        #img = Image.open(os.path.join(root_dir, image_name))
        #img = np.array(img.convert('RGB'))[:, :, ::-1].copy()
        cairosvg.svg2png(url=os.path.join(root_dir, image_name),
                write_to=image_name.replace(".svg", ".png"))
        img = cv2.imread(image_name.replace(".svg", ".png"))
        os.system("rm % s" % image_name.replace(".svg", ".png"))

    img = np.array(img)
    if img.ndim < 3:
        img = np.tile(np.expand_dims(img, 2), (1, 1, 3))
    img = cv2.resize(img, (56, 56))
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    img -= np.array([0.485, 0.456, 0.406]).astype('float32').reshape(3, 1, 1)
    img /= np.array([0.229, 0.224, 0.225]).astype('float32').reshape(3, 1, 1)

    return img

def get_result(model, image_name, args):
    img = get_img(image_name)
    input = torch.from_numpy(img).float()
    
    input = input.unsqueeze(0)
    input = input.to(device=args.dev)
    output = model(input)
    soft = torch.nn.Softmax()
    output = soft(output)
    score, pred = output.topk(1, 1, True, True)
    return pred[0][0], score[0][0]

def main(args):
    model = models.get_model(load_pretrain=False).to(device=args.dev)
    checkpoint = torch.load(args.checkpoint)

    model.load_state_dict(checkpoint) 
    model.eval()

    test_filenames = []
    for line in open("./data/test_files.txt"):
        test_filenames.append(line.strip().split()[0])

    fp = open(args.save_file, 'w')
    for test_fn in tqdm(test_filenames):
        try:
            pred, score = get_result(model, test_fn, args)
            fp.writelines("%s %d %f\n" % (test_fn, pred, score))
        except:
            print(test_fn)
    fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fake icon rec test')
    parser.add_argument('--dev', default='cpu', type=str,
            help='testing device')
    parser.add_argument('--checkpoint', default='checkpoint', type=str,
            help='test checkpoint')
    parser.add_argument('--save-file', default='test_results.txt', type=str,
            help='test results')
    args = parser.parse_args()
    main(args)
