import random
import pandas as pd
from sklearn import metrics, model_selection

path_label = "/data/game/cancer/data/train_label.csv"

df = pd.read_csv(path_label)
kf = model_selection.StratifiedKFold(n_splits=5, random_state = 2020, shuffle=True)

df_result = None
for i, (t, v) in enumerate(kf.split(df['image_name'], df['label'])):
    df_val = df.loc[df.index.isin(v)]
    df_val.insert(2, "fold", i)
    if i == 0:
        df_result = df_val
    else:
        df_result = pd.concat([df_result, df_val])
    print(len(t), len(v))

df_result.to_csv(f"validation_data.csv", index=False)