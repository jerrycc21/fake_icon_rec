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

def get_img(image_path):
    ext = image_path.split(".")[-1]
    if ext in ['png', 'jpg']:
        img = cv2.imread(image_path)
    elif ext == 'gif':
        t = imageio.mimread(image_path, memtest=False)
        if len(t[0].shape) < 3:
            img = t[0][:, :]
        else:
            img = t[0][:, :, :3]
    elif ext == 'svg':
        #img = Image.open(os.path.join(root_dir, image_name))
        #img = np.array(img.convert('RGB'))[:, :, ::-1].copy()
        cairosvg.svg2png(url=image_path,
                write_to=image_path.replace(".svg", ".png"))
        img = cv2.imread(image_path.replace(".svg", ".png"))
        os.system("rm % s" % image_path.replace(".svg", ".png"))

    img = np.array(img)
    if img.ndim < 3:
        img = np.tile(np.expand_dims(img, 2), (1, 1, 3))
    img = cv2.resize(img, (56, 56))
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    img -= np.array([0.485, 0.456, 0.406]).astype('float32').reshape(3, 1, 1)
    img /= np.array([0.229, 0.224, 0.225]).astype('float32').reshape(3, 1, 1)

    return img

def get_result(model, args):
    img = get_img(args.image_path)
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

    pred, score = get_result(model, args)
    if pred == 0:
        #print("fake icon: ", pred.data.numpy(), score.data.numpy())
        print("fake icon")
    else: 
        #print("True icon: ", pred.data.numpy(), score.data.numpy())
        print("Real icon")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fake icon rec test')
    parser.add_argument('--dev', default='cpu', type=str,
            help='testing device')
    parser.add_argument('--image-path', default='./test.png', type=str,
            help='testing image path')
    parser.add_argument('--checkpoint', default='./checkpoint', type=str,
            help='testing checkpoint')
    args = parser.parse_args()
    main(args)
