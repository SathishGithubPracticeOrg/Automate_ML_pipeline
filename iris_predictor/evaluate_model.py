
import pickle
import pandas as pd
#import openpyxl
import json

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import Config
#Config.METRICS_FILE_PATH.mkdir(parents = True, exist_ok= True)

X_test = pd.read_csv(str(Config.FEATURES_PATH/"test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH/"test_labels.csv"))

model = pickle.load(open(str(Config.MODELS_PATH /"model.pickle"), "rb"))
y_pred = model.predict (X_test)

metrics_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
print(metrics_report)

accuracy = metrics.accuracy_score (y_test, y_pred)
precision = metrics.precision_score (y_test, y_pred, average = None)
f1 = metrics.f1_score (y_test, y_pred, average = None)
recall = metrics.recall_score (y_test, y_pred, average = None)

print("Accuracy Score = ", accuracy, \
    "Precision score = " ,precision, \
        "F1 score = ",f1, \
            "Recall score = ", recall)

with open(str(Config.ASSETS_PATH/ "metrics.json"), "w") as outfile:
    json.dump(dict(accuracy = accuracy), outfile)