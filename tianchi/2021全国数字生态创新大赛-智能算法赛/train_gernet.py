import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import cv2, gc

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms as T

from tqdm import tqdm
from glob import glob

from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp

import torch.utils.data as D
from torch.utils.data import Dataset, DataLoader

from focalDiceLoss import FocalDiceLoss

from torch.autograd import Variable

# 配置
images_dir = './images/'
mask_dir = './masks/'

model_name = 'timm-gernet_l'
n_class = 10
batch_size = 32
EPOCHS = 100
DEVICE = "cuda:0"


# 设置随机，以便结果一致
def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seeds();

# 数据增强
tifm = A.Compose([
    A.Resize(256, 256),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5)
    ]),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 数据dataset
class EcologicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = os.path.join(self.imgs_dir, idx + '.jpg')
        mask_file = os.path.join(self.masks_dir, idx + '.png')

        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        return image, mask.long()

# 网络结构
class Model(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.model = smp.Unet(
                encoder_name = model_name,
                encoder_weights = "imagenet",
                in_channels = 3,
                classes = n_class,
            )
    def forward(self, x):
        x = self.model(x)
        return x

@torch.no_grad()
def validation(model, loader, loss_fn):
    val_iou = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output = output.argmax(1)
        iou = np_iou(output, target)
        val_iou.append(iou)

    return val_iou

def np_iou(pred, mask, c=10):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)

        uion = p.sum() + t.sum()
        overlap = (p*t).sum()

        iou = 2*overlap/(p.sum() + t.sum() +0.001)
        iou_result.append(iou.abs().data.cpu().numpy())

    return np.stack(iou_result)


header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'
print(header)
class_name = ['farm','land','forest','grass','road','urban_area',
                 'countryside','industrial_land','construction',
                 'water', 'bareland']

print('  '.join(class_name))

all_data = EcologicDataset(images_dir, mask_dir, transform=tifm)
ss = KFold(n_splits=5, shuffle=True, random_state=1)

for fold_idx, (train_index, val_index) in enumerate(ss.split(all_data)):
    train_data = D.Subset(all_data, train_index)
    valid_data = D.Subset(all_data, val_index)
    print(len(train_data), len(valid_data))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Model(model_name, n_class).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    criterion = FocalDiceLoss()

    best_iou = 0.0
    for epoch in range(1, EPOCHS+1):
        losses = []
        start_time = time.time()
        model.train()
        for batch_idx, (image,target) in tqdm(enumerate(train_loader)):
            image, target = Variable(image.to(DEVICE)), Variable(target.to(DEVICE))
            with torch.set_grad_enabled(True):
                pred = model(image)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                optimizer.zero_grad()

        viou = validation(model, valid_loader, criterion)
        print('\t'.join(np.stack(viou).mean(0).round(3).astype(str)))

        print(raw_line.format(epoch, np.array(losses).mean(), np.mean(viou),
                                    (time.time()-start_time)/60**1))

        if best_iou < np.stack(viou).mean(0).round(3)[-1]:
            best_iou = np.stack(viou).mean(0).round(3)[-1]
            torch.save(model.state_dict(), 'train_timm-gernet_l.pth')
    break
