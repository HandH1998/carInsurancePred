import numpy as np
import pandas as pd
import joblib
import time
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from utils.dataprocess import preProc, preProcTest, toJson, sampleWeight
import sklearn.metrics as metrics

t1 = time.time()
dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
weight_negative = sum(dataTrain['Response']) * 1.0 / dataTrain.shape[0]
xTrain, yTrain, scaler = preProc(dataTrain)
sample_weight = sampleWeight(yTrain, weight_negative)

# model=MultinomialNB()
model = GaussianNB()
# 赋上sample权重
model.fit(xTrain, yTrain, sample_weight=sample_weight)
score = model.score(xTrain, yTrain)
print('Score:', score)
pred_y = model.predict(xTrain)
fscore = metrics.f1_score(yTrain, pred_y)
print('Fvalue:', fscore)
equal1count = sum(pred_y == 1)
print('预测结果为1的数量:', equal1count)
print(pred_y)
joblib.dump(model, 'carInsurancePredNB.model')

# 预测
model = joblib.load('carInsurancePredNB.model')
dataTest = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest = preProcTest(dataTest,scaler)
model = joblib.load('carInsurancePredSVM.model')
pred_y = model.predict(xTest)
equal1count = sum(pred_y == 1)
print('预测结果为1的数量:', equal1count)
toJson(pred_y)