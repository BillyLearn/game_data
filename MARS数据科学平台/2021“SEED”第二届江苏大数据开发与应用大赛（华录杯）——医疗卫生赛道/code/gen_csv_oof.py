import random
import pandas as pd

# path_label = "./volo_log/oof_df.csv"
# df = pd.read_csv(path_label)
# # path_label = "./医学模型/oof_df.csv"
# df = pd.read_csv(path_label)


# df_label_pred = df.loc[df['label'] == df['preds']]
# print("total ", len(df_label_pred))
# print("\n")

# df_label_0 = df_label_pred.loc[df['label'] == 0]
# df_label_00 = df_label_0.loc[df['0'] > 0.99]
# print("label 0 ", len(df_label_00), " 小于0.99 删 ", len(df_label_0) - len(df_label_00))
# print("\n")

# df_label_1 = df_label_pred.loc[df['label'] == 1]
# df_label_11 = df_label_1.loc[df['1'] > 0.99]
# print("label 1 ", len(df_label_11), " 小于0.99 删 ", len(df_label_1) - len(df_label_11))
# print("\n")

# df_label_2 = df_label_pred.loc[df['label'] == 2]
# df_label_22 = df_label_2.loc[df['2'] > 0.99]
# print("label 2 ", len(df_label_22), " 小于0.99 删 ",  len(df_label_2) - len(df_label_22))
# print("\n")

# df_label_3 = df_label_pred.loc[df['label'] == 3]
# df_label_33 = df_label_3.loc[df['3'] > 0.7]
# print("label 3 ", len(df_label_33), " 小于0.99 删 ", len(df_label_3) - len(df_label_33))
# print("\n")

# df_label_0 = df_label_0

# df_all = pd.concat([df_label_00, df_label_11,df_label_22,df_label_33],axis=0,ignore_index=True)
# path_label2 = "./oof_label.csv"
# df2 = pd.read_csv(path_label2)
# df_all = pd.concat([df_label_00, df_label_11,df_label_22, df2],axis=0,ignore_index=True)
# df_all.to_csv('oof_label_1.csv', index=False)
# print("all do ", len(df_all))


# path_label = "./result/result.csv"
# path_label = "oof_label.csv"
# df = pd.read_csv(path_label)

# ids0 = df[df['label'] == 0].image_name.values
# ids1 = df[df['label'] == 1].image_name.values
# ids2 = df[df['label'] == 2].image_name.values
# ids3 = df[df['label'] == 3].image_name.values
# print("label 0 ", len(ids0))
# print("label 1 ", len(ids1))
# print("label 2 ", len(ids2))
# print("label 3 ", len(ids3))


# path_label = "oof_label.csv"
# df = pd.read_csv(path_label)

# df = pd.read_csv("./result/result.csv")

# df_label_0 = df.loc[df['label'] == 0]
# df_label_00 = df_label_0.loc[df['preds'] > 0.9]
# print("label 0 ", len(df_label_00), " 小于0.99 删 ", len(df_label_0) - len(df_label_00))
# print("\n")

# df_label_1 = df.loc[df['label'] == 1]
# df_label_11 = df_label_1.loc[df['preds'] > 0.9]
# print("label 1 ", len(df_label_11), " 小于0.99 删 ", len(df_label_1) - len(df_label_11))
# print("\n")

# df_label_2 = df.loc[df['label'] == 2]
# df_label_22 = df_label_2.loc[df['preds'] > 0.9]
# print("label 2 ", len(df_label_22), " 小于0.99 删 ", len(df_label_2) - len(df_label_22))
# print("\n")

# df_label_3 = df.loc[df['label'] == 3]
# df_label_33 = df_label_3.loc[df['preds'] > 0.8]
# print("label 3 ", len(df_label_33), " 小于0.99 删 ", len(df_label_3) - len(df_label_33))
# print("\n")

# df_all = pd.concat([df_label_00, df_label_11, df_label_22, df_label_33],axis=0,ignore_index=True)
# df_all.to_csv('oof_test_label.csv', index=False)

# df_label_2 = df_label_pred.loc[df['label'] == 2]
# df_label_22 = df_label_2.loc[df['2'] > 0.99]
# print("label 2 ", len(df_label_22), " 小于0.99 删 ",  len(df_label_2) - len(df_label_22))
# print("\n")


# path_label = "oof_label.csv"
# df1 = pd.read_csv(path_label)
# df2 = pd.read_csv("oof_test_label.csv")

# df3 = pd.read_csv("/data/game/cancer/data/train_label.csv")
# df3_label_3 = df3.loc[df3['label'] == 3]
# print(len(df3_label_3))

# df_all = pd.concat([df1, df2, df3_label_3],axis=0,ignore_index=True)
# df_all.to_csv('oof_all.csv', index=False)
# ids0 = df_all[df_all['label'] == 0].image_name.values
# ids1 = df_all[df_all['label'] == 1].image_name.values
# ids2 = df_all[df_all['label'] == 2].image_name.values
# ids3 = df_all[df_all['label'] == 3].image_name.values
# print("label 0 ", len(ids0))
# print("label 1 ", len(ids1))
# print("label 2 ", len(ids2))
# print("label 3 ", len(ids3))



LABELS = 'validation_data.csv'
label = 0
fold = 4

df = pd.read_csv(LABELS)
ids = df[(df['label'] == label) & (df['fold'] == fold)].image_name.values
print(len(ids))