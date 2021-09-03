from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import os
import cv2
import gc
import random
import albumentations as A
from lovasz import lovasz_hinge

import torch
import torch.utils.data as D
import segmentation_models_pytorch as smp
import torch.nn as nn
import time

import logging

nfolds = 5
SEED = 2020
EPOCHES = 25 
model_name = "resnext101_320_320_320"
if not os.path.exists(model_name):
    os.makedirs(model_name)

TRAIN = '/data/game/cancer/data/train_origin_image'
MASKS = '/data/game/cancer/data/train_mask'
LABELS = 'validation_data.csv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_WORKERS = 0
logging.basicConfig(filename=f'{model_name}/log.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

## dataset
def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class CancerDataset(D.Dataset):
    def __init__(self, img_size, fold, train=True, tfms=None):
        df = pd.read_csv(LABELS)
        if train:
            ids = df[(df['label'] == label) & (df['fold'] != fold)].image_name.values
        else:
            ids = df[(df['label'] == label) & (df['fold'] == fold)].image_name.values

        self.fnames = [fname for fname in os.listdir(TRAIN) if fname in ids]
        self.train = train
        self.tfms = tfms
        self.img_size = img_size

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN,fname)), cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
        mask = np.divide(mask, 255.0)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation = cv2.INTER_AREA)

        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
        return img2tensor((img/255.0 - mean)/std),img2tensor(mask)

def get_aug(p=1.0):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                        border_mode=cv2.BORDER_REFLECT),
        A.OneOf([
            A.ElasticTransform(p=.3),
            A.GaussianBlur(p=.3),
            A.GaussNoise(p=.3),
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(15,25,0),
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        ], p=0.3),
        A.OneOf([
            A.CoarseDropout(p=0.3),
            A.Cutout(p=0.3),
        ], p=0.3),
    ], p=p)

# def get_model():
#     model = smp.DeepLabV3Plus(
#         encoder_name="resnext101_32x8d",
#         encoder_weights="None",
#         in_channels=3,
#         classes=1
#     )
#     return model

# from network import UneXt50
# def get_model():
#     model = UneXt50(2)
#     return model

from network import UneXt101
def get_model():
    model = UneXt101(2)
    return model

## N表示画面中总共像素的个数；TP表示正确预测的病灶区域的像素个数；TN表示正确预测的健康区域的像素个数；
def AP(pred, mask):
    p = pred.reshape(-1)
    t = mask.reshape(-1)

    N = len(p)
    TP = (p*t).sum()
    TN = N - TP - (abs(p-t).sum())
    return (TP+TN) / N

def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))

def train(model, train_loader, optimizer):
    losses = []
    torch.cuda.empty_cache()
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = symmetric_lovasz(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        torch.cuda.empty_cache()

    return np.array(losses).mean()

def validation(model, val_loader):
    val_probability, val_mask = [], []
    model.eval()
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            output = model(image)

            output_ny = output.sigmoid().data.cpu().numpy()
            target_np = target.data.cpu().numpy()

            val_probability.append(output_ny)
            val_mask.append(target_np)

    val_probability = np.concatenate(val_probability)
    val_mask = np.concatenate(val_mask)

    return AP(val_probability, val_mask)

for label in [1, 2, 3]:
    img_size = 320
    bs = 8

    mean = np.array([0.485, 0.456, 0.406]),
    std =  np.array([0.229, 0.224, 0.225]),

    save_path = f'{model_name}/{label}_cancer'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seed_everything(SEED)

    # 训练
    for fold in range(nfolds):
        train_ds = CancerDataset(img_size, fold=fold, train=True, tfms=get_aug())
        valid_ds = CancerDataset(img_size, fold=fold, train=False)

        train_loader = D.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

        val_loader = D.DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=0, drop_last=True)
        print(len(train_loader), len(val_loader))

        model = get_model()
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        header = r'''
                Train | Valid
        Epoch |  Loss |  AP (Best) | Time
        '''
        print(header)
        #          Epoch         metrics            time
        raw_line = '{:6d}' + '\u2502{:7.4f}'*3 + '\u2502{:6.2f}'

        best_ap = 0
        for epoch in range(1, EPOCHES+1):
            start_time = time.time()
            model.train()
            train_loss = train(model, train_loader, optimizer)
            val_ap = validation(model, val_loader)
            lr_step.step(val_ap)

            if val_ap >= best_ap:
                best_ap = val_ap
                torch.save(model.state_dict(), f'{save_path}/fold_{fold}.pth')

            print(raw_line.format(epoch, train_loss, val_ap, best_ap, (time.time()-start_time)/60**1))
            logging.info(raw_line.format(epoch, train_loss, val_ap, best_ap, (time.time()-start_time)/60**1))


        del train_loader, val_loader, train_ds, valid_ds
        gc.collect();
