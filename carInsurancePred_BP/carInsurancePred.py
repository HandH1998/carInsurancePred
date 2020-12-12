import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

def preProc(dataTain):
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
    print(xTrain.columns)
    # rus = RandomUnderSampler(random_state=0)
    # xTrain, yTrain = rus.fit_sample(xTrain, yTrain)
    # print(xTrain.head(10))
    # xTrain.boxplot(fontsize=6)
    # plt.show()
    xTrain = torch.from_numpy(np.array(xTrain)).type(torch.FloatTensor)
    yTrain = torch.from_numpy(np.array(yTrain)).type(torch.LongTensor)
    return xTrain, yTrain


dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
response = dataTrain['Response'].value_counts()
print(response[1] * 1.0 / sum(response))
weights=[response[1] * 1.0 / sum(response),response[0] * 1.0 / sum(response)]
weights=torch.from_numpy(np.array(weights)).type(torch.FloatTensor)
# threshold = response[1] * 1.0 / response[0]
xTrain, yTrain = preProc(dataTrain)
# print(dataTrain.columns)
print(xTrain, yTrain)
torch.save(xTrain, 'xTrain.pt')
torch.save(yTrain, 'yTrain.pt')
BATCH_SIZE = 5000
EPOCH = 50
torch_dataset = Data.TensorDataset(xTrain, yTrain)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        # self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        # self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        x = self.predict(x)
        return x


def accuracyCal(y1, y2):
    return sum(y1 == y2) / y1.shape[0]


def precisionCal(y1, y2):
    return sum(np.all([y1 == 1, y1 == y2], axis=0)) * 1.0 / sum(y1 == 1)


def recallCal(y1, y2):
    return sum(np.all([y1 == 1, y1 == y2], axis=0)) * 1.0 / sum(y2 == 1)


# def fscoreCal(y1, y2):
def fscoreCal(p, r, b):
    return ((1 + b ** 2) * p * r) / ((p * b ** 2) + r)
    # tp = sum(np.all([y1 == 1, y1 == y2], axis=0))
    # tn = sum(np.all([y1 == 0, y1 == y2], axis = 0))
    # return 2*tp*1.0/(y1.shape[0]+tp-tn)


net = Net(14, 10, 2)
print(net)
LR = 0.02
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
# optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.8)
# optimizer = torch.optim.SGD(net.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss(weight=weights)
plt.ion()
fscoreList = []
epochList = []
lossList = []
for epoch in range(EPOCH):
    print('epoch:', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        output = net(batch_x)
        # print(torch.argmax())
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prob = F.softmax(output, dim=1)
    # probUpdate = torch.cat((prob[:, 0].unsqueeze(-1) * threshold, prob[:, 1].unsqueeze(-1)), -1)
    # print(prob[:,1].size())
    # print(prob[:,1].unsqueeze(-1).size())
    # print(probUpdate)
    # print(probUpdate.shape)
    # prediction = torch.max(probUpdate, 1)[1]
    prediction=torch.max(prob,1)[1]
    pred_y = prediction.numpy()
    target_y = batch_y.numpy()
    # print(pred_y.shape[0])
    # accuracy=sum(pred_y==target_y)/BATCH_SIZE
    # print('accuracy:',accuracy)
    p = precisionCal(pred_y, target_y)
    r = recallCal(pred_y, target_y)
    # fscore = fscoreCal(pred_y, target_y)
    fscore = fscoreCal(p, r, 1)
    print('预测结果为1的数量：', sum(pred_y == 1))
    print('Fvalue:', fscore)
    print('Precision:', p)
    print('Recall:', r)
    epochList.append(epoch)
    fscoreList.append(fscore)

    plt.plot(epochList, fscoreList)
    plt.xlabel(u'迭代次数')
    plt.ylabel('Fvalue')
    plt.title(u'训练过程')

    # print('loss:',loss.detach().numpy())
    # lossList.append(loss)
    # plt.plot(epochList, lossList)
    # plt.xlabel(u'迭代次数')
    # plt.ylabel(u'误差')
    # plt.title(u'误差训练过程')
    plt.pause(0.2)

plt.ioff()
plt.show()

torch.save(net, 'carInsurancePred.pt')
torch.save(net.state_dict(), 'carInsurancePredParams.pt')
# torch.save(threshold,'threshold.pt')


