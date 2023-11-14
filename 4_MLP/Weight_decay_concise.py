import torch as tf
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn
import math

class Accumulator:
    """
    累加器
    """
    def __init__(self,n):
        self.data=[0.0 for i in range(n)]

    def add(self,*args):#累加
        self.data=[a+float(b) for a,b in zip(self.data,args)]

    def reset(self):#归零
        self.data=[0.0 for i in range(len(self.data))]

    def __getitem__(self,idx):
        return self.data[idx]

def synthetic_data(w,b,data_cnt):#(权重向量，常数项，样本数量)
    """
    生成 y = Xw + b + 噪声 的数据
    """
    x=tf.normal(0,1,(data_cnt,len(w)))
    y=tf.matmul(x,w.reshape((-1,1)))+b
    y+=tf.normal(0,0.01,y.shape)
    return x,y

def load_array(data_arrays,batch_size,is_train=True):
    """
    生成pytorch数据迭代器
    """
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

def squared_loss(y_hat,y):#(预测值，真实值)
    """
    损失函数
    """
    return (y_hat-y.reshape(y_hat.shape))**2/2



def evaluate_loss(net, data_iter, loss):
    """评估模型在指定数据集上的损失"""
    m = Accumulator(2)
    for X,y in data_iter:
        y_hat = net(X)
        l=loss(y_hat,y.reshape(y_hat.shape))
        m.add(l.sum(),l.numel())
    return m[0]/m[1]

def train_concise(lambd,epoch_cnt,lr):
    net = nn.Sequential(nn.Linear(ip_cnt,1))
    loss = nn.MSELoss(reduction='none')
    for param in net.parameters():
        param.data.normal_()
    trainer = tf.optim.SGD([{"params":net[0].weight,"weight_decay":lambd},{"params":net[0].bias}],lr)
    
    train_loss_lst=[]
    test_loss_lst=[]
    for epoch in range(epoch_cnt):
        for X,y in train_iter:
            trainer.zero_grad()
            l = loss(net(X),y)
            l.mean().backward()
            trainer.step()

        train_loss,test_loss=[evaluate_loss(net,train_iter,loss),evaluate_loss(net,test_iter,loss)]
        train_loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)
        print("epoch: %d train loss: %f test_loss: %f"%(epoch,train_loss,test_loss))
    print('w的L2范数是：',net[0].weight.norm().item())
    plt.figure()
    train_loss_lst=[math.log10(x) for x in train_loss_lst]
    test_loss_lst=[math.log10(x) for x in test_loss_lst]
    plt.plot(list(range(1,epoch_cnt+1)),train_loss_lst,label='log10(train)')
    plt.plot(list(range(1,epoch_cnt+1)),test_loss_lst,label='log10(test)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_cnt, test_cnt, ip_cnt, batch_size=20,100,200,5


    w0=tf.ones((ip_cnt,1))*0.01
    b0=0.05

    train_data=synthetic_data(w0,b0,train_cnt)
    test_data=synthetic_data(w0,b0,test_cnt)

    train_iter=load_array(train_data,batch_size)
    test_iter=load_array(test_data,batch_size)

    train_concise(lambd=3,epoch_cnt=100,lr=0.003)