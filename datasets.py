import os
import cv2
import random
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class ICONTRAIN(Dataset):
    def __init__(self, scale_size=56):
        super(ICONTRAIN, self).__init__()
        self.root_dir = "./data/"
        self.img_root_dir = os.path.join(self.root_dir, 'icons')
        self.scale_size = scale_size
        self.train_files_path = os.path.join(self.root_dir, "train_files.txt")
        self.train_real_fake_files_path = os.path.join(self.root_dir,
                "train_real_fake_files.txt")
        self.train_files = get_filenames(self.train_files_path)
        self.train_real_fake_files = get_filenames(self.train_real_fake_files_path)
        self.marker_file = os.path.join(self.root_dir, "verify_small.png")
        #self.marker_file1 = os.path.join(self.root_dir, "verify.png")

    def __getitem__(self, idx):
        ## real sample
        if random.random() < 0.5:
            fn = self.train_files[idx]
            target = 1
            img = self.get_image(fn)
            #if random.random() < 0.05:
            #    img[:, :, 0] *= 1.5
        else:
            ## real fake
            if random.random() < 0.06:
                fn = random.choice(self.train_real_fake_files)
                target = 0
                img = self.get_image(fn) 
            ## fake 
            else:
                fn = self.train_files[idx]
                target = 0
                img = self.generate_fake_img(fn)
        img = cv2.resize(img, (self.scale_size, self.scale_size))
        img = self.transform(img)
        img = torch.from_numpy(img).float()
        return img, target

    def transform(self, img):
        img = np.array(img)
        if img.ndim < 3:
            img = np.tile(np.expand_dims(img, 2), (1, 1, 3))
        img = img.transpose((2, 0, 1))
        img = self.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img

    def normalize(self, img, mean, std):
        img = img / 255.0
        img -= np.array(mean).astype('float32').reshape(3, 1, 1)
        img /= np.array(std).astype('float32').reshape(3, 1, 1)
        return img

    def generate_fake_img(self, filename):
        img = self.get_image(filename)
        if len(img.shape) < 3:
            img = np.tile(np.expand_dims(img, 2), (1, 1, 3))
        #if random.random() < 0.3:
        #    marker = cv2.imread(self.marker_file1)
        #else:
        marker = cv2.imread(self.marker_file)
        h, w, c = img.shape
        h_s = h // 3 * 2
        w_s = w // 3 * 2
        new_marker_size = min(h, w) // 5
        mask = np.zeros((h, w, c), dtype=np.uint8)
        try:
            new_marker = cv2.resize(marker, (new_marker_size, new_marker_size))
        except:
            print(filename)
        mask[h_s:h_s+new_marker_size, w_s:w_s+new_marker_size] = new_marker
        t = mask.copy()
        mask[mask > 0] = 1
        mask = 1 - mask
        fake_img = img * mask + t
        return fake_img

    def get_image(self, filename):
        if filename.split(".")[1] in ["png", "jpg"]:
            img = cv2.imread(os.path.join(self.img_root_dir, filename))
        else:
            ## gif
            t = imageio.mimread(os.path.join(self.img_root_dir, filename),
                    memtest=False)
            if len(t[0].shape) < 3:
                img = t[0][:, :]
            else:
                img = t[0][:, :, :3]
        return img

    def __len__(self):
        return len(self.train_files)

def get_filenames(meta_path):
    filenames = []
    for line in open(meta_path):
        filenames.append(line.strip())
    return filenames
