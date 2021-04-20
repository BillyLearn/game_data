import os
import cv2
import numpy as np
import onnxruntime as rt

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torchvision import transforms as T
import segmentation_models_pytorch as smp

def image_process(image_path):
    mean = np.array([[[0.485, 0.456, 0.406]]])      # 训练的时候用来mean和std
    std = np.array([[[0.229, 0.224, 0.225]]])

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))                 # (96, 96, 3)

    image = img.astype(np.float32)/255.0
    image = (image - mean)/ std

    image = image.transpose((2, 0, 1))              # (3, 96, 96)
    image = image[np.newaxis,:,:,:]                 # (1, 3, 96, 96)

    image = np.array(image, dtype=np.float32)

    return image

def onnx_runtime():
    imgdata = image_process('/data/game/tianchi/suichang/compete_datasets/test_jpg/001746.jpg')

    sess = rt.InferenceSession('se_resnext101.onnx')
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    pred_onnx = sess.run([output_name], {input_name: imgdata})

    pred_onnx = np.array(pred_onnx).squeeze()
    pred = np.argmax(pred_onnx, axis=0)
    print("outputs:", type(pred), np.shape(pred))

onnx_runtime()

class SegSuiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
                encoder_name = "se_resnext101_32x4d",
                encoder_weights = None,
                in_channels = 3,
                classes = 10,
            )
    def forward(self, x):
        x = self.model(x)
        return x

def get_infer_transform():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform

img_dir = '/data/game/tianchi/suichang/compete_datasets/test_jpg/001746.jpg'
image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = SegSuiModel().to(device)
model.load_state_dict(torch.load(f'/data/game/tianchi/suichang/new_project/train_file/se_resnext101.pth'))
model.eval()

transform = get_infer_transform()
# img = transform(image=image)['image']
img = transform(image)
img = img.unsqueeze(0)
img = img.to(device)
output = model(img)

pred = output.squeeze().cpu().data.numpy()
print("1: ", type(pred), np.shape(pred))
pred = np.argmax(pred, axis=0)
print("2: ", np.shape(pred))
