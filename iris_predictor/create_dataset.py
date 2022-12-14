import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from config import Config

np.random.seed(Config.RANDOM_SEED)

Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents= True, exist_ok= True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv ("https://raw.githubusercontent.com/insaid2018/Term-3/master/Projects/Iris.csv")
df.to_csv (str(Config.ORIGINAL_DATASET_FILE_PATH), index = False)

df_train, df_test = train_test_split(df, test_size = 0.2, random_state = Config.RANDOM_SEED,stratify = df.Species )

df_train.to_csv(str(Config.DATASET_PATH/ "train.csv"), index = None)
df_test.to_csv(str(Config.DATASET_PATH/ "test.csv"), index = None)