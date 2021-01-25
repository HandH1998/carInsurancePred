from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
import sklearn.tree as tree
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
import joblib
from utils.dataprocess import preProc, preProcTest, toJson

dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
xTrain, yTrain, scaler = preProc(dataTrain)
model_tree = tree.DecisionTreeClassifier(criterion='gini', class_weight='balanced', random_state=0, max_depth=8,
                                         min_impurity_decrease=0.01, min_samples_leaf=6)
# 以决策树为基模型的Adaboost
model = AdaBoostClassifier(base_estimator=model_tree, algorithm='SAMME', n_estimators=500, learning_rate=1)
model.fit(xTrain, yTrain)
pred_y = model.predict(xTrain)
fscore = metrics.f1_score(yTrain, pred_y)
print('Fvalue:', fscore)
joblib.dump(model, 'carInsurancePredAdaboost.model')

# 预测
model = joblib.load('carInsurancePredAdaboost.model')
dataTest = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest = preProcTest(dataTest,scaler)
pred_y = model.predict(xTest)
countEqual1 = sum(pred_y == 1)
print('预测结果为1的数量:', countEqual1)
toJson(pred_y)
