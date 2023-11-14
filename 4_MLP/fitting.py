import math
import random
import numpy as np
import torch as tf
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

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

def syn_data():
    """生成多项式数据集"""
    features = np.random.normal(size=(n_train+n_test,1))
    np.random.shuffle(features)
    poly_feature = np.power(features,np.arange(max_degree).reshape((1,-1)))
    for i in range(max_degree):
        poly_feature[:,i]/=math.gamma(i+1)
    labels = np.dot(poly_feature,w0.reshape((-1,1)))
    labels+=np.random.normal(scale=0.1,size=labels.shape)
    return features,poly_feature,labels.reshape((-1,))

def load_array(data_arrays,batch_size,is_train=True):
    """
    生成pytorch数据迭代器
    """
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

def evaluate_loss(net, data_iter, loss):
    """评估模型在指定数据集上的损失"""
    m = Accumulator(2)
    for X,y in data_iter:
        y_hat = net(X)
        l=loss(y_hat,y.reshape(y_hat.shape))
        m.add(l.sum(),l.numel())
    return m[0]/m[1]

def train_epoch(net,train_iter,loss,updater):
    """
    训练模型时的单个迭代周期
    """
    if isinstance(net,tf.nn.Module):
        net.train()#将模型设置为训练模式（可计算梯度）
    m = Accumulator(3)#累加器，（损失总和，预测正确样本数，样本总数）

    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)

        #使用pytorch内置优化器：
        if isinstance(updater,tf.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()

        #自定义优化器：
        else:
            l.sum().backward()
            updater(X.shape[0])#即batch_size

        m.add(float(l.sum()),y.numel())
    return m[0]/m[1]

def train(train_features, test_features, train_labels, test_labels, epoch_cnt=400):
    """训练"""
    loss = nn.MSELoss(reduction='none')
    """
    正确的写法就是loss = nn.MSELoss(reduction=‘none’)，
    这样求出来的是【张量】，如果不指定reduction则默认是均值（标量）。

    新版的train_epoch_ch3函数接受的loss就是一个张量，
    函数里通过l.mean().backward()求出均值【标量】以进行梯度计算。

    上古版本的train_epoch_ch3代码错误的将loss梯度计算处理为l.backward(),
    则要求接受的loss就是一个标量，
    所以需要使用loss的均值即 loss = nn.MSELoss()，
    否则标量和张量格式不匹配无法进行梯度运算，
    才会报错RuntimeError: grad can be implicitly created only for scalar outputs 。
    这与当时使用的优化器是pytorch内置还是d2l自己实现的有关，
    pytorch内置优化器求均值是在loss计算中完成的，d2l自己定义的优化器求均值是在sgd时完成的。
    一个要求的loss是均值（标量），一个要求的loss是求和（l.sum(),l是张量)，
    所以各种写法只要对应上，基本也不会差太多（只要标量和张量别搞错，就算mean和sum搞错，
    最多就差一个batch_size量级learning rate）。
    """
    ip_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(ip_shape,1,bias=False))
    batch_size = min(10,train_labels.shape[0])

    train_iter = load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = load_array((test_features,test_labels.reshape(-1,1)),batch_size,is_train=False)
    trainer = tf.optim.SGD(net.parameters(),lr=0.1)

    train_loss_lst=[]
    test_loss_lst=[]
    for epoch in range(epoch_cnt):
        l=train_epoch(net,train_iter,loss,trainer)
        train_loss_lst.append(evaluate_loss(net,train_iter,loss))
        test_loss_lst.append(evaluate_loss(net,test_iter,loss))
        print("epoch: %d, train loss: %f, test_loss: %f"%(epoch+1,train_loss_lst[-1],test_loss_lst[-1]))
    return train_loss_lst,test_loss_lst

if __name__=='__main__':
    max_degree = 20  # 多项式的最大阶数
    n_train, n_test = 100, 100  # 训练和测试数据集大小
    poly_cnt=20#拟合多项式次数

    w0=np.zeros(max_degree)
    w0[:4]=np.array([5,1.2,-3.4,5.6])
    features,poly_features,labels=syn_data()
    # print(features[:2])
    # print(poly_features[:2])
    # print(labels[:2])
    w0,features,poly_features,labels=[tf.tensor(x,dtype=tf.float32) for x in[w0,features,poly_features,labels]]

    # print(poly_features[:n_train, :4])
    # print(labels[:n_train])

    train_loss_lst,test_loss_lst=train(poly_features[:n_train, :poly_cnt], poly_features[n_train:, :poly_cnt],
                                        labels[:n_train], labels[n_train:])


    train_loss_lst=[math.log10(x) for x in train_loss_lst]
    test_loss_lst=[math.log10(x) for x in test_loss_lst]

    plt.figure()
    plt.plot(list(range(1,401)),train_loss_lst,label='log10(train)')
    plt.plot(list(range(1,401)),test_loss_lst,label='log10(test)')
    plt.legend()
    plt.show()
