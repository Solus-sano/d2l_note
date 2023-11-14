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

def init_params():
    """初始化模型参数"""
    w=tf.normal(0,1,size=(ip_cnt,1),requires_grad=True)
    b=tf.zeros(1,requires_grad=True)
    return [w,b]

def l2_penatly(w):
    """L2罚函数"""
    return tf.sum(w.pow(2))/2

def linreg(x,w,b):
    """高维线性模型"""
    return tf.matmul(x,w.reshape((-1,1)))+b

def sgd(params,lr,batch_size):#(参数(包含w、b)，学习率，批量大小(用于梯度下降步长规范化))
    """
    小批量随机梯度下降
    """
    with tf.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()#梯度归零

def evaluate_loss(net, data_iter, loss,w,b):
    """评估模型在指定数据集上的损失"""
    m = Accumulator(2)
    for X,y in data_iter:
        y_hat = net(X,w,b)
        l=loss(y_hat,y.reshape(y_hat.shape))
        m.add(l.sum(),l.numel())
    return m[0]/m[1]

def train(lambd,epoch_cnt,lr):
    w,b=init_params()
    net = linreg
    loss = nn.MSELoss()
    train_loss_lst=[]
    test_loss_lst=[]
    for epoch in range(epoch_cnt):
        for X,y in train_iter:
            l = loss(net(X,w,b),y) + lambd*l2_penatly(w)
            l.sum().backward()
            sgd((w,b),lr,batch_size)
        train_loss,test_loss=[evaluate_loss(net,train_iter,loss,w,b),evaluate_loss(net,test_iter,loss,w,b)]
        train_loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)
        print("epoch: %d train loss: %f test_loss: %f"%(epoch,train_loss,test_loss))
    print('w的L2范数是：', tf.norm(w).item())
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

    train(lambd=20,epoch_cnt=400,lr=0.003)