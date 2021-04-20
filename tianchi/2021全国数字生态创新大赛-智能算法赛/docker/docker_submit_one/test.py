from ai_hub import inferServer
import json
import base64
import cv2
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torchvision import transforms as T

import sys
sys.path.append("./seg_pytorch")
import segmentation_models_pytorch as smp

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


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


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # 数据前处理
    def pre_process(self, data):
        #json process
        json_data = json.loads(data.get_data().decode('utf-8'))
        img = json_data.get("img")
        bast64_data = img.encode(encoding='utf-8')
        img = base64.b64decode(bast64_data)
        colorImg = cv2.imdecode(np.frombuffer(bytearray(img)), cv2.IMREAD_COLOR)
        rgbImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
        img = self.as_tensor(rgbImg)
        img = img.unsqueeze(0)
        return img

    # 数据后处理
    def post_process(self, data):
        processed_data = np.argmax(data, axis=0) + 1
        img_encode = np.array(cv2.imencode('.png', processed_data)[1]).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data,'utf-8')
        return bast64_str

    #模型预测：默认执行self.model(preprocess_data)，一般不用重写
    #如需自定义，可覆盖重写
    def predict(self, data):
        with torch.no_grad():
            img = data.to(device)
            output = self.model(img)
            pred = output.squeeze().cpu().data.numpy()
        return pred

if __name__ == "__main__":
    model = SegSuiModel().to(device)
    model.load_state_dict(torch.load(f'timm-gernet_l.pth'))
    model.eval()
    my_infer = myInfer(model)
    my_infer.run(debuge=True) #默认为("127.0.0.1", 80)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息
