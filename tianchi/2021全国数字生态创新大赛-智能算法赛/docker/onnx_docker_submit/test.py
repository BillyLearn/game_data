from ai_hub import inferServer
import json
import base64
import cv2
from io import BytesIO
import numpy as np
import onnxruntime as rt

class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    # 数据前处理
    def pre_process(self, data):
        #json process
        json_data = json.loads(data.get_data().decode('utf-8'))
        img = json_data.get("img")
        bast64_data = img.encode(encoding='utf-8')
        img = base64.b64decode(bast64_data)
        colorImg = cv2.imdecode(np.frombuffer(bytearray(img)), cv2.IMREAD_COLOR)

        rgbImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgbImg, (256, 256))

        mean = np.array([[[0.485, 0.456, 0.406]]])
        std = np.array([[[0.229, 0.224, 0.225]]])
        image = img.astype(np.float32)/255.0
        image = (image - mean)/ std
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis,:,:,:]
        image = np.array(image, dtype=np.float32)
        return image

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
        pred_onnx = self.model.run([self.output_name], {self.input_name: data})
        pred_onnx = np.array(pred_onnx).squeeze()
        return pred_onnx

if __name__ == "__main__":
    sess = rt.InferenceSession('gernet_l.onnx')
    my_infer = myInfer(sess)
    my_infer.run(debuge=True) #默认为("127.0.0.1", 80)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息
