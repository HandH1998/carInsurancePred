import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import joblib
from utils.dataprocess import preProc, preProcTest, toJson
from sklearn.model_selection import train_test_split, GridSearchCV

# xgb sklearn接口


dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
xTrain, yTrain, scaler = preProc(dataTrain, toNumpy=True)
# weight=反例数量/正例数量
weight = (yTrain.shape[0] - sum(yTrain)) * 1.0 / sum(yTrain)

# xTrain,xValidation,yTrain,yValidation=train_test_split(xTrain,yTrain,test_size=0.2)
# xgb.XGBClassifier参数与xgb.train有区别
params = {
    # xgboost宏观特征参数
    'booster': 'gbtree',
    'nthread': 5,  # 线程数
    # 'silent': 0,  # 为1时，静默开启
    'n_estimators': 350,  # change! 相当于num_boost_round

    # booster参数
    'learning_rate': 0.1,  # change!通过减少每一步的权重，提高鲁棒性
    'gamma': 0.1,  # 节点分裂所需要的最小损失函数下降值
    'max_depth': 9,  # 最大树高，限制过拟合
    'reg_lambda': 2,  # chagne! 权重的L2正则项
    'reg_alpha': 1,  # change! 权重的L1正则项
    'subsample': 0.9,  # 这个参数控制对于每棵树，随机采样的比例，控制过拟合
    'colsample_bytree': 0.6,  # 用来控制每棵随机采样的列数的占比(每一列是一个特征)，控制过拟合
    'min_child_weight': 3,  # 决定最小叶子节点样本权重和，较大时，可以避免学到局部特殊样本
    'scale_pos_weight': weight,  # 样本类别不平衡时设置，常设置为：反例数量/正例数量

    # 学习目标参数
    'objective': 'binary:logistic',  # 这是定义的需要最小化的损失函数， 回归任务设置为：'objective': 'reg:gamma',
    # 'num_class': 3,  # 多分类设置
    'eval_metric': 'logloss',  # 评估指标
    # 'seed': 1000,  # 随机数种子，可以复现随机数据的结果,，也可以用于调整参数
}
model = xgb.XGBClassifier(**params)  # **params 以字典形式传入的参数解包，XGBClassifier的参数是直接放在构造里边的，与train不一样

# 网格调优
# cv_params = {'n_estimators': [i for i in range(100, 1100, 50)]} # n_estimators=350
# cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]} # max_depth=9,min_child_weight=3
# cv_params = {'gamma': [i for i in np.linspace(0.1,0.8,8)]} # gamma=0.1
# cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]} # subsample=0.9,colsample_bytree=0.6
# cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]} # reg_alpha=1,reg_lambda=2
# cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]} # learning_rate=0.1
# cv_params = {'n_estimators': [i for i in range(100, 1100, 50)]} # n_estimators=350
# optimized_model = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, n_jobs=4,
#                                verbose=0)  # cv 交叉验证
# optimized_model.fit(xTrain, yTrain)
# evaluate_result = optimized_model.cv_results_
# print('result: ', evaluate_result)
# print('best_params: ', optimized_model.best_params_)
# print('best_score: ', optimized_model.best_score_)

model.fit(xTrain,yTrain)
pred_y=model.predict(xTrain)
pred_y = np.array([round(pred) for pred in pred_y])
fscore=metrics.f1_score(yTrain,pred_y)
print('Fvalue:',fscore)
joblib.dump(model, 'carInsurancePredXgboost.model')

# 预测
model = joblib.load('carInsurancePredXgboost.model')
dataTest = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest = preProcTest(dataTest,scaler,toNumpy=True)
pred_y = model.predict(xTest)
pred_y = np.array([round(pred) for pred in pred_y])
countEqual1 = sum(pred_y)
print('预测结果为1的数量:', countEqual1)
toJson(pred_y)
