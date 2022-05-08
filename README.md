## Fake Icon Recognition

#### Environment
The code is developed using Python 3.7.9 on Centos 7.0. Other platforms are not fully tested. You can run this code on CPU or GPU.

#### Installation
1. Clone this repo

```
git clone https://github.com/jerrycc21/fake_icon_rec.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

#### Data Preparation
Download the data, the icon images will stored in `./data/icons/`
```
cd data
python download_data.py
```
The image names corresponding to the training and testing sets are stored in `./data/train_files.txt`, `./data/train_real_fake_files.txt`, and `/data/test_files.txt` respectively.

#### Demo

We provide pretrained model parameters that you can test on a single image.
```
python test.py --checkpoint checkpoints/fake_icon_model.pth --image-path test.jpg --dev cpu
```

#### Training
Train your own model on the CPU, and the model parameters are stored in the `./checkpoints` folder.
```
python train.py --epochs 50 --batch-size 64 --lr 0.001 --dev cpu --dump-dir ./checkpoints
```
Or train your model on GPU
```
python train.py --epochs 50 --batch-size 64 --lr 0.001 --dev cuda --dump-dir ./checkpoints
```

#### Testing
Test on the full test set
```
python test.py --checkpoint checkpoints/fake_icon_model.pth --save-file test_results.txt --dev cpu
```
Calculate the accuracy
```
python cal_acc.py --gt ./data/test_files.txt --pred ./test_results.txt
```



#### Acknowledgement
Thanks to [x2y2](https://x2y2.io) for the data support!


