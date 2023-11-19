import torch
import numpy as np
from sklearn import datasets
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
dataset = datasets.load_iris()
x=dataset.data
y=dataset.target

input, x_test,label, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

input = torch.FloatTensor(input)
label = torch.LongTensor(label)
x_test =torch.FloatTensor(x_test)
y_test =torch.LongTensor(y_test)

label_size = int(np.array(label.size()))

class NET(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(NET, self).__init__()
        self.hidden1 =nn.Linear(n_feature,n_hidden1)
        self.relu1 =nn.ReLU()

        self.hidden2 =nn.Linear(n_hidden1,n_hidden2)
        self.relu2 =nn.ReLU()

        self.out =nn.Linear(n_hidden2,n_output)
        self.softmax =nn.LogSoftmax(dim=1)
#前向传播函数
    def forward(self, x):
        hidden1 = self.hidden1(x)
        relu1 = self.relu1(hidden1)
        hidden2 =self.hidden2(relu1)
        relu2 = self.relu2(hidden2)

        out =self.out(relu2)

        return out
#测试函数
    def test(self, x):
        y_pred = self.forward(x)
        y_predict = self.softmax(y_pred)

        return y_predict


# 定义网络结构以及损失函数

net = NET(n_feature= input.shape[1], n_hidden1=40, n_hidden2=20, n_output=len(np.unique(label)))
optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()
costs = []
# 训练网络
for epoch in range(2000):
    cost = 0
    out = net(input)
    loss = loss_func(out,label)
    optimizer.zero_grad()
# 反向传播 并更新所有参数
    loss.backward()
    optimizer.step()
    cost = cost + loss.cpu().detach().numpy()
    costs.append(cost / label_size)
#可视化
plt.plot(costs)
plt.show()

# 测试训练集准确率
out = net.test(input)
prediction = torch.max(out, 1)[1]
pred_y = prediction.numpy()
target_y = label.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("Training set accuracy:", accuracy * 100, "%")

# 测试测试集准确率
out1 = net.test(x_test)
prediction1 = torch.max(out1, 1)[1]
pred_y1 = prediction1.numpy()
target_y1 = y_test.numpy()

accuracy1 = float((pred_y1 == target_y1).astype(int).sum()) / float(target_y1.size)
print("Testing set accuracy:", accuracy1 * 100, "%")

