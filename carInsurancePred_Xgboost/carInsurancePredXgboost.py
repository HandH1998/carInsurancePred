import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import joblib
from utils.dataprocess import preProc, preProcTest, toJson
from sklearn.model_selection import train_test_split, GridSearchCV

# xgb原生接口

dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
xTrain, yTrain, scaler = preProc(dataTrain, toNumpy=True)
# weight=反例数量/正例数量
weight = (yTrain.shape[0] - sum(yTrain)) * 1.0 / sum(yTrain)
xTrain, xValidation, yTrain, yValidation = train_test_split(xTrain, yTrain, test_size=0.2)

dtrain = xgb.DMatrix(xTrain, label=yTrain)
dvalidation = xgb.DMatrix(xValidation, label=yValidation)
params = {
    # xgboost宏观特征参数
    'booster': 'gbtree',
    'nthread': 5,  # 线程数
    'silent': 0,  # 为1时，静默开启

    # booster参数
    'eta': 0.1,  # learning rate 通过减少每一步的权重，提高鲁棒性
    'gamma': 0.1,  # 节点分裂所需要的最小损失函数下降值
    'max_depth': 9,  # 最大树高，限制过拟合
    'lambda': 2,  # 权重的L2正则项
    'alpha': 1,  # 权重的L1正则项
    'subsample': 0.9,  # 这个参数控制对于每棵树，随机采样的比例，控制过拟合
    'colsample_bytree': 0.6,  # 用来控制每棵随机采样的列数的占比(每一列是一个特征)，控制过拟合
    'min_child_weight': 3,  # 决定最小叶子节点样本权重和，较大时，可以避免学到局部特殊样本
    # 'seed': 1000,
    'scale_pos_weight': weight,  # 样本类别不平衡时设置，常设置为：反例数量/正例数量

    # 学习目标参数
    'objective': 'binary:logistic',  # 这是定义的需要最小化的损失函数， 回归任务设置为：'objective': 'reg:gamma',
    # 'num_class': 3,  # 多分类设置
    'eval_metric': 'logloss',  # 评估指标
    'seed': 1000,  # 随机数种子，可以复现随机数据的结果,，也可以用于调整参数
}
watch_list = [(dtrain, 'train'), (dvalidation, 'validation')]
# res 为交叉验证之后的各个树的结果，res.shape[0]即为最优数量
res = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
# print('最优次数 ', res.shape[0])
model = xgb.train(params, dtrain, num_boost_round=res.shape[0], evals=watch_list, verbose_eval=True,
                  early_stopping_rounds=500)
# plt.ion()
# xgb.plot_tree(model,num_trees=0)
# plt.show()
pred_y = model.predict(dtrain)
pred_y = np.array([round(pred) for pred in pred_y])
fscore = metrics.f1_score(yTrain, pred_y)
print('Fvalue:', fscore)
joblib.dump(model, 'carInsurancePredXgboost.model')



# 预测
model = joblib.load('carInsurancePredXgboost.model')
dataTest = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest = preProcTest(dataTest, scaler, toNumpy=True)
dtest = xgb.DMatrix(xTest)
pred_y = model.predict(dtest)
pred_y = np.array([round(pred) for pred in pred_y])
print(pred_y)
countEqual1 = sum(pred_y)
print('预测结果为1的数量:', countEqual1)
toJson(pred_y)
