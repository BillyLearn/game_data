import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

path = '/data/game/cancer/data/train_origin_image'
df_train = pd.read_csv("/data/game/cancer/data/train_label.csv")

image_names = []
labels = []
widths = []
heights = []
x_tot = []
x2_tot = []

for label in [1, 2, 3]:
    ids = df_train[df_train['label'] == label].image_name.values
    for i, image_name in tqdm(enumerate(ids)):
        img_path = os.path.join(path, image_name)
        im = cv2.imread(img_path)
        h,w,_ = im.shape

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x_tot.append((im/255.0).reshape(-1,3).mean(0))
        x2_tot.append(((im/255.0)**2).reshape(-1,3).mean(0))

        image_names.append(image_name)
        labels.append(label)
        widths.append(w)
        heights.append(h)

    #image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
    print('label ', label, 'mean:',img_avr, ', std:', img_std)

# 生成新的csv文件
data_dict = {"image_name": image_names,
             "label": labels,
            "width": widths,
            "height":heights
           }

df = pd.DataFrame(data_dict)
df.to_csv (r'new_train_label.csv', index = False, header=True)


df_train = pd.read_csv("./new_train_label.csv")

sources = df_train['label'].unique()
print(f"标签总共有 {len(sources)} 类 {sources}")
print("\n")
source = df_train['label'].value_counts()
print(source)

for label in [1, 2, 3]:
    print("\n")
    width = df_train[df_train['label'] == label].width.values
    hight = df_train[df_train['label'] == label].height.values
    print("label: ", label, " min width",  min(width), " max width ", max(width))
    print("label: ", label, " min height",  min(hight), " max height", max(hight))
