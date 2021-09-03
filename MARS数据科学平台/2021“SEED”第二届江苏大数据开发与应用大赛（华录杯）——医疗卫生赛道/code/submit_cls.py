import cv2
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, ShiftScaleRotate, RandomBrightness, RandomContrast, RandomGamma, CenterCrop, Blur, VerticalFlip, HueSaturationValue, RandomRotate90,RandomResizedCrop, Transpose, RandomBrightnessContrast, CoarseDropout, Cutout

from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data
import timm
from tqdm import tqdm
import torch.nn.functional as F
from models import volo_d2
from utils import load_pretrained_weights as volo_load_weights
import ttach as tta


path = '/data/game/cancer/data/test_image'
result_path = "./result"
if not os.path.exists(result_path):
    os.makedirs(result_path)

classes = 4

image_id = list(os.listdir(path))
print(len(image_id))

def get_inference_transforms(img_size):
    return Compose([
            Resize(int(img_size), int(img_size)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_model_volo():
    base_model = timm.create_model("volo_d2", img_size=224, mix_token=False, return_dense=False)
    in_features = base_model.head.in_features
    base_model.head = nn.Linear(in_features=in_features, out_features=classes, bias=True)
    return base_model

def get_model_nfnet():
    model = timm.create_model('dm_nfnet_f3', pretrained=False)
    global_pool = model.head.global_pool
    n_features = model.head.fc.in_features

    model.head.global_pool = global_pool
    model.head.fc = nn.Linear(n_features, classes)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net0 = []
for i in [0,1,2,3,4]:
    model = get_model_volo()
    state_dict = torch.load(f"volo_log/volo_d2_fold{str(i)}_best.pth")['model']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    net0.append(model)

for i in [0,1,2,3]:
    model = get_model_nfnet()
    state_dict = torch.load(f"./医学模型/dm_nfnet_f3_cls_lsloss_img224_cleandata/nfnet_log/dm_nfnet_f3_fold{str(i)}_best.pth")['model']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    net0.append(model)

print(len(net0))


# 预测
probability = []
preds = []

for file in tqdm(image_id):
    img_path = os.path.join(path, file)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = get_inference_transforms(224)(image=image)
    image1 = augmented['image'][None]
    image1 = image1.to(device)

    image_preds_all = []

    with torch.no_grad():
        for model in net0:
            tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform())
            image_preds = model(image1)
            image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)

    tta_preds = np.mean(image_preds_all, axis=0)
    prob = np.argmax(tta_preds)
    preds.append(tta_preds[prob])
    probability.append((prob))

df_submit = pd.DataFrame({'image_name': image_id, 'label': probability})
df_submit.to_csv(f'{result_path}/result.csv', index=False)
