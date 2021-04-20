import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torchvision import transforms as T

import segmentation_models_pytorch as smp

class SegSuiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
                encoder_name = "timm-gernet_l",
                encoder_weights = None,
                in_channels = 3,
                classes = 10,
            )
    def forward(self, x):
        x = self.model(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


model = SegSuiModel().to(device)
model.load_state_dict(torch.load(f'/data/game/tianchi/suichang/new_project/train_file/train_best_model_0.pth'))
print(model)
model.eval()

batch_size = 1  #批处理大小
input_shape = (3, 256, 256)   #输入数据,改成自己的输入shape

x = torch.randn(batch_size, *input_shape)
x = x.to(device)
export_onnx_file = "gernet_l.onnx"		#输出的ONNX文件名

torch.onnx.export(model,
                x,
                export_onnx_file,
                export_params=True,
                verbose=True,
                input_names=["input"],
                output_names=["output"])
