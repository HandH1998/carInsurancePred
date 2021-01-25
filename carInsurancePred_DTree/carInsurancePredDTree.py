import numpy as np
import pandas as pd
import sklearn.tree as tree
import sklearn.metrics as metrics
import joblib
from utils.dataprocess import preProc, preProcTest, toJson

# 处理训练数据
dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
xTrain, yTrain, scaler = preProc(dataTrain)
# 建立模型，训练
model = tree.DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=0, max_depth=8,
                                    min_impurity_decrease=0.01, min_samples_leaf=4)# class_weight='balanced' 调整类别不均衡
model.fit(xTrain, yTrain)
pred_y = model.predict(xTrain)
fscore = metrics.f1_score(yTrain, pred_y)
print('Fvalue:', fscore)
joblib.dump(model, 'carInsurancePredDTree.model')

# 预测
model = joblib.load('carInsurancePredDTree.model')
dataTest = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest = preProcTest(dataTest,scaler)
pred_y = model.predict(xTest)
countEqual1 = sum(pred_y == 1)
print('预测结果为1的数量:', countEqual1)
toJson(pred_y)
