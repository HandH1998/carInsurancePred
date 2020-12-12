from sklearn.svm import SVC
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt


def preProc(dataTrain):
    dataTrain.dropna(axis=0, how='any', inplace=True)
    dataTrain.drop(dataTrain.columns[0:2], axis=1, inplace=True)
    yTrain = dataTrain.iloc[:, -1]
    xTrain = dataTrain.iloc[:, :-1]
    # print(yTrain)
    # print(xTrain)
    xTrain = pd.get_dummies(xTrain, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'])
    temp = xTrain.pop('Driving_License')
    xTrain.insert(13, 'Driving_License', temp)
    temp = xTrain.pop('Previously_Insured')
    xTrain.insert(13, 'Previously_Insured', temp)
    xTrain = pd.concat([xTrain.iloc[:, :5].apply(lambda x: (x - np.min(x)) / np.std(x)), xTrain.iloc[:, 5:]], axis=1)
    # print(xTrain.columns)
    # print(xTrain.head(10))
    # xTrain.boxplot(fontsize=6)
    # plt.show()
    # xTrain = torch.from_numpy(np.array(xTrain)).type(torch.FloatTensor)
    # yTrain = torch.from_numpy(np.array(yTrain)).type(torch.LongTensor)
    return xTrain, yTrain





def fscoreCal(y1, y2):
    # def fscoreCal(p, r, b):
    #     return ((1 + b ** 2) * p * r) / ((p * b ** 2) + r)
    tp = sum(np.all([y1 == 1, y1 == y2], axis=0))
    tn = sum(np.all([y1 == 0, y1 == y2], axis=0))
    return 2 * tp * 1.0 / (y1.shape[0] + tp - tn)


t1 = time.time()
dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
# response = dataTrain['Response'].value_counts()
# threshold = response[1] * 1.0 / response[0]
xTrain, yTrain = preProc(dataTrain)

max_iter = 100000000
model = SVC(C=1.0, kernel='rbf', gamma='auto', tol=0.2, cache_size=1024, class_weight='balanced', max_iter=max_iter)
model.fit(xTrain, yTrain)
score = model.score(xTrain, yTrain)
print('Score:', score)
pred_y = model.predict(xTrain)
fscore = fscoreCal(pred_y, yTrain)
print('Fvalue:', fscore)
equal1count = sum(pred_y == 1)
print('预测结果为1的数量:', equal1count)
print(pred_y)
joblib.dump(model, 'carInsurancePredSVM.model')

print('迭代次数', max_iter, '耗时:', time.time() - t1)

# model=joblib.load('carInsurancePredSVM.model')
# pred_y = model.predict(xTrain)
# print('保存的model预测结果:',pred_y)
