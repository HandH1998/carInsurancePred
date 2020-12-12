import json
from collections import OrderedDict
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt


def preProcTest(dataTest):
    dataTest.dropna(axis=0, how='any', inplace=True)
    xTest = dataTest.drop(dataTest.columns[0], axis=1)
    # print(yTest)
    # print(xTest)
    xTest = pd.get_dummies(xTest, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'])
    temp = xTest.pop('Driving_License')
    xTest.insert(13, 'Driving_License', temp)
    temp = xTest.pop('Previously_Insured')
    xTest.insert(13, 'Previously_Insured', temp)
    xTest = pd.concat([xTest.iloc[:, :5].apply(lambda x: (x - np.min(x)) / np.std(x)), xTest.iloc[:, 5:]], axis=1)
    print(xTest.columns)
    print(xTest.head(10))
    # xTest.boxplot(fontsize=6)
    # plt.show()
    # xTest = torch.from_numpy(np.array(xTest)).type(torch.FloatTensor)
    return xTest


def fscoreCal(y1, y2):
    # def fscoreCal(p, r, b):
    #     return ((1 + b ** 2) * p * r) / ((p * b ** 2) + r)
    tp = sum(np.all([y1 == 1, y1 == y2], axis=0))
    tn = sum(np.all([y1 == 0, y1 == y2], axis=0))
    return 2 * tp * 1.0 / (y1.shape[0] + tp - tn)


dataTest = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest = preProcTest(dataTest)
model = joblib.load('carInsurancePredSVM.model')
pred_y = model.predict(xTest)
equal1count = sum(pred_y == 1)
print('预测结果为1的数量:', equal1count)
dic = OrderedDict()
for i, res in enumerate(pred_y):
    dic[str(i)] = res.tolist()

jsonStr = json.dumps(dic, indent=4)
with open('submission.json', 'w') as f:
    f.write(jsonStr)