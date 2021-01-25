from sklearn.svm import SVC
import numpy as np
import pandas as pd
import joblib
import time
from utils.dataprocess import preProc, preProcTest, toJson
import sklearn.metrics as metrics

t1 = time.time()
# 训练数据处理
dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
xTrain, yTrain, scaler = preProc(dataTrain)

# 建立模型
max_iter = 100000000
# model里加上class_weight='balanced',等价于正负例分别乘以权重sum(负例)、sum(正例)
# fit里有参数sample_weight,为每个sample赋上权重，是长度等于sample数量的array
# 这两个作用相同，只使用一个
model = SVC(C=1.0, kernel='rbf', gamma='auto', tol=0.2, cache_size=1024, class_weight='balanced', max_iter=max_iter)
model.fit(xTrain, yTrain, sample_weight=None)
score = model.score(xTrain, yTrain)
print('Score:', score)
pred_y = model.predict(xTrain)
fscore = metrics.f1_score(yTrain, pred_y)
print('Fvalue:', fscore)
equal1count = sum(pred_y == 1)
print('预测结果为1的数量:', equal1count)
print(pred_y)
joblib.dump(model, 'carInsurancePredSVM.model')
print('迭代次数', max_iter, '耗时:', time.time() - t1)

# 预测
model = joblib.load('carInsurancePredSVM.model')
dataTest = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest = preProcTest(dataTest, scaler)
pred_y = model.predict(xTest)
equal1count = sum(pred_y == 1)
print('预测结果为1的数量:', equal1count)
toJson(pred_y)
