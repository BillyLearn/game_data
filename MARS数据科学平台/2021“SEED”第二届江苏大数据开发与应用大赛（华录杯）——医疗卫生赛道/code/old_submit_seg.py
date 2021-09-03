import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.utils.data as D
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import time
from tqdm import tqdm

DATA = '/data/game/cancer/data/test_image'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

result_path = "./result/result"
df = pd.read_csv("./result/result.csv")

if not os.path.exists(result_path):
    os.makedirs(result_path)

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

## model
def get_model():
    model = smp.DeepLabV3Plus(
        encoder_name="resnext101_32x8d",
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    return model

from network_old import UneXt101
def get_model_unext101():
    model = UneXt101()
    return model

from network import UneXt50
def get_model_unext50():
    model = UneXt50(2)
    return model

# class 0
ids = df[df['label'] == 0].image_name.values
for i, image_name in tqdm(enumerate(ids)):
    ori_img = cv2.imread(os.path.join(DATA,image_name))
    ori_h, ori_w, _ = ori_img.shape
    mask = np.zeros((ori_h, ori_w), dtype='uint8')
    cv2.imwrite(os.path.join(result_path, image_name), mask)

for label in [1, 2, 3]:
    if label == 1:
        thr = 0.5
        model_name = ""
        MODELS = [f'/data/game/cancer/submit_all/医学模型/resnext101_32x8d_320/{label}_cancer/fold_{i}.pth' for i in range(4)]
        MODELS2 = [f'/data/game/cancer/submit_all/医学模型/DeepLabV3/DeepLabV3Plus_resnext101_32x8d_20210801_size320_bs16/{label}_cancer/fold_{i}.pth' for i in range(4)]
        mean = np.array([0.70838886, 0.43378238, 0.69586983])
        std = np.array([0.19031998, 0.25258623, 0.14332357])
        img_size = 320

    if label == 2:
        thr = 0.5
        MODELS = [f'/data/game/cancer/submit_all/医学模型/resnext101_32x8d_320/{label}_cancer/fold_{i}.pth' for i in range(4)]
        MODELS2 = [f'/data/game/cancer/submit_all/医学模型/DeepLabV3/DeepLabV3Plus_resnext101_32x8d_20210801_size320_bs16/{label}_cancer/fold_{i}.pth' for i in range(4)]
        mean = np.array([0.80871184, 0.64832798, 0.80332749])
        std = np.array([0.18473279, 0.2788455, 0.15412724])
        img_size = 320

    if label == 3:
        thr = 0.6
        mean = np.array([0.485, 0.456, 0.406]),
        std =  np.array([0.229, 0.224, 0.225]),
        MODELS3 = [f'./resnext50_320_320_320_copy/{label}_cancer/fold_{i}.pth' for i in range(5)]
        img_size = 320

    ids = df[df['label'] == label].image_name.values

    models = []

    if label == 1 or label==2:
        for path in MODELS:
            state_dict = torch.load(path)
            model = get_model_unext101()
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            models.append(model)

        for path in MODELS2:
            state_dict = torch.load(path)
            model = get_model()
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            models.append(model)

    if label == 3:
        for path in MODELS3:
            state_dict = torch.load(path)
            model = get_model_unext50()
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            models.append(model)

    print(len(models))

    for i, image_name in tqdm(enumerate(ids)):
        ori_img = cv2.cvtColor(cv2.imread(os.path.join(DATA,image_name)), cv2.COLOR_BGR2RGB)
        ori_h, ori_w, _ = ori_img.shape

        img = cv2.resize(ori_img, (img_size, img_size), interpolation = cv2.INTER_AREA)
        img = img2tensor((img/255.0 - mean)/std)
        img = img.to(device)[None]

        py = None
        with torch.no_grad():
            for model in models:
                p = model(img)
                p = torch.sigmoid(p).detach()
                if py is None: py = p
                else: py += p

            # x,y,xy flips as TTA
            flips = [[-1],[-2],[-2,-1]]
            for f in flips:
                xf = torch.flip(img,f)
                for model in models:
                    p = model(xf)
                    p = torch.flip(p,f)
                    py += torch.sigmoid(p).detach()

        py /= (1+len(flips))
        py /= len(models)
        py = F.interpolate(py, scale_factor=2, mode='bilinear')
        py = py.permute(0,2,3,1).float().cpu().numpy()
        mask = cv2.resize(py[0], (ori_w, ori_h))
        ret,mask = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(result_path, image_name), mask)
