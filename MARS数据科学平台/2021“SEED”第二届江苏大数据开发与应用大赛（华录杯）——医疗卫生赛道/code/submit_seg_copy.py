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

from network import UneXt50
# def get_model_unext50():
def get_model():
    model = UneXt50(2)
    return model

# from network import UneXt101
# def get_model():
#     model = UneXt101()
#     return model

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
        # MODELS = [f'./resnext101_320_320_320/{label}_cancer/fold_{i}.pth' for i in range(4)]
        MODELS = [f'./resnext50_320_320_320_copy/{label}_cancer/fold_{i}.pth' for i in range(4)]
        # mean = np.array([0.69805722, 0.42196582, 0.68982128])
        # std = np.array([0.24647701, 0.29372352, 0.18023622])
        mean = np.array([0.69337369, 0.41660915, 0.68707925])
        std = np.array([0.24754964, 0.29179386, 0.18005593])
        img_size = 320

    if label == 2:
        thr = 0.5
        # MODELS = [f'./resnext101_320_320_320/{label}_cancer/fold_{i}.pth' for i in range(4)]
        MODELS = [f'./resnext50_320_320_320_copy/{label}_cancer/fold_{i}.pth' for i in range(4)]
        # mean = np.array([0.75267474, 0.53769139, 0.74688154])
        # std = np.array([0.23190909, 0.31414159, 0.18215843])
        mean = np.array([0.73594514, 0.50885032, 0.7316511])
        std = np.array([0.23817769, 0.31365463, 0.1837299 ])
        img_size = 320

    if label == 3:
        thr = 0.5
        # MODELS = [f'./resnext101_320_320_320/{label}_cancer/fold_{i}.pth' for i in range(4)]
        MODELS = [f'./resnext50_320_320_320_copy/{label}_cancer/fold_{i}.pth' for i in range(4)]
        # mean = np.array([0.7447854, 0.52139289, 0.73751313])
        # std = np.array([0.23464427, 0.31511629, 0.18410069])
        mean = np.array([0.72839049, 0.49166157, 0.72139842])
        std = np.array([0.23998517, 0.31353449, 0.1853373])
        img_size = 320

    ids = df[df['label'] == label].image_name.values

    models = []

    for path in MODELS:
        state_dict = torch.load(path)
        model = get_model()
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
        py = F.interpolate(py, scale_factor=2, mode='bilinear', align_corners=True)
        py = py.permute(0,2,3,1).float().cpu().numpy()
        mask = cv2.resize(py[0], (ori_w, ori_h))
        ret,mask = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(result_path, image_name), mask)
