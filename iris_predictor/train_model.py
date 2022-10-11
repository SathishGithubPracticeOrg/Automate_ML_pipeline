import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import Config

Config.MODELS_PATH.mkdir(parents = True, exist_ok = True)

X_train = pd.read_csv(str(Config.FEATURES_PATH/"train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH/"train_labels.csv"))

model =RandomForestClassifier(n_estimators=100,n_jobs=3)
model = model.fit(X_train, y_train.to_numpy().ravel())

pickle.dump(model, open(str(Config.MODELS_PATH/"model.pickle"), "wb"))