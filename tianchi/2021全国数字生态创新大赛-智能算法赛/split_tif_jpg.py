import cv2.cv2 as cv
import cv2
import os
import shutil
from tqdm import tqdm

val_images_dir='/data/game/tianchi/suichang/compete_datasets/suichang_round1_train_210120'
# val_save_imgs='/data/game/tianchi/suichang/new_project/val/images'
# val_save_masks='/data/game/tianchi/suichang/new_project/val/masks'
# if not os.path.exists(val_save_imgs):os.makedirs(val_save_imgs)
# if not os.path.exists(val_save_masks):os.makedirs(val_save_masks)

train_images_dir='/data/game/tianchi/suichang/new_project/suichang_round2_train_210316/'
train_save_imgs='/data/game/tianchi/suichang/new_project/train/images'
train_save_masks='/data/game/tianchi/suichang/new_project/train/masks'
if not os.path.exists(train_save_imgs):os.makedirs(train_save_imgs)
if not os.path.exists(train_save_masks):os.makedirs(train_save_masks)



# train_tif_list = [x for x in os.listdir(train_images_dir)]   # 获取目录中所有tif格式图像列表
# for num,name in tqdm(enumerate(train_tif_list)):      # 遍历列表
#     if name.endswith(".tif"):
#         img = cv.imread(os.path.join(train_images_dir, name),-1)       #  读取列表中的tif图像
#         cv.imwrite(os.path.join(train_save_imgs,name.split('.')[0]+".jpg"),img)    # tif 格式转 jpg
#     else:
#         img = cv.imread(os.path.join(train_images_dir, name),cv2.IMREAD_GRAYSCALE)
#         img = img-1
#         cv2.imwrite(os.path.join(train_save_masks, name),img)


# val_tif_list = [x for x in os.listdir(val_images_dir)]
# for num,name in tqdm(enumerate(val_tif_list)):      # 遍历列表
#     if name.endswith(".tif"):
#         img = cv.imread(os.path.join(val_images_dir, name),-1)       #  读取列表中的tif图像
#         cv.imwrite(os.path.join(val_save_imgs,name.split('.')[0]+".jpg"),img)    # tif 格式转 jpg
#     else:
#         img = cv.imread(os.path.join(val_images_dir, name),cv2.IMREAD_GRAYSCALE)
#         img = img-1
#         cv2.imwrite(os.path.join(val_save_masks, name),img)

val_tif_list = [x for x in os.listdir(val_images_dir)]
for num,name in tqdm(enumerate(val_tif_list)):      # 遍历列表
    if name.endswith(".tif"):
        img = cv.imread(os.path.join(val_images_dir, name),-1)       #  读取列表中的tif图像
        name = "1_" + name
        print("name", name)
        cv.imwrite(os.path.join(train_save_imgs,name.split('.')[0]+".jpg"),img)    # tif 格式转 jpg
    else:
        img = cv.imread(os.path.join(val_images_dir, name),cv2.IMREAD_GRAYSCALE)
        img = img-1
        name = "1_" + name
        print("name", name)
        cv2.imwrite(os.path.join(train_save_masks, name),img)