import torchvision
from torch.utils import data
from torchvision import transforms
import torch as tf
from torch import nn
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

class Net(nn.Module):
    def __init__(self,ip_cnt,op_cnt,hid1_cnt,hid2_cnt,p1,p2,is_training=True):
        super(Net,self).__init__()
        self.ip_cnt=ip_cnt
        self.training=is_training
        self.lin1=nn.Linear(ip_cnt,hid1_cnt)
        self.lin2=nn.Linear(hid1_cnt,hid2_cnt)
        self.lin3=nn.Linear(hid2_cnt,op_cnt)
        self.relu=nn.ReLU()
        self.p1=p1
        self.p2=p2

    def forward(self,x):
        h1=self.relu(self.lin1(x.reshape(-1,self.ip_cnt)))

        if self.training==True:
            h1=dropout_layer(h1,self.p1)
        
        h2=self.relu(self.lin2(h1))

        if self.training==True:
            h2=dropout_layer(h2,self.p2)

        h3=self.relu(self.lin3(h2))
        return h3

def load_data(batch_size,resize=None,worker_cnt=4):
    """
    下载数据集，并将其加入到内存中，生成迭代器并返回
    """
    trans=[transforms.ToTensor()]
    if resize:
        trans.append(transforms.Resize(resize))
    trans=transforms.Compose(trans)

    #下载训练数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data",
        train=True,
        transform=trans,
        download=True
    )

    #下载验证数据集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data",
        train=False,
        transform=trans,
        download=True
    )

    return (data.DataLoader(mnist_train,batch_size,shuffle=True,
                            num_workers=worker_cnt),
            data.DataLoader(mnist_test,batch_size,shuffle=True,
                            num_workers=worker_cnt))

def dropout_layer(x,p):
    """以概率p去除x中元素"""
    assert 0<=p<=1
    if p==1:
        return tf.zeros_like(x)
    if p==0:
        return x
    mask =  (tf.randn(x.shape)>p).float()
    return mask*x/(1.0-p)

def accuracy(y_hat,y):
    """
    计算一个batch的预测精度
    """
    if y_hat.shape[0]>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)#取概率最大的下标作为预测类别
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())#预测正确的样本数

def evaluate_accuracy(net, data_iter):
    """
    评估模型在指定数据集上的精度
    """
    if isinstance(net,tf.nn.Module):
        net.eval()#将模型设置为评估模式
    m = Accumulator(2)
    with tf.no_grad():
        for X,y in data_iter:
            m.add(accuracy(net(X),y),y.numel())
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

        m.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return m[0]/m[2] , m[1]/m[2]

def train(net, train_iter, test_iter, loss, epoch_cnt, updater):
    # anim = Animator(xlabel='epoch', xlim=[1, epoch_cnt], ylim=[0.3, 0.9],
    #                 legend=['train loss', 'train acc', 'test acc'])
    train_loss_lst=[]
    train_accuracy_lst=[]
    test_accuracy_lst=[]
    for epoch in range(epoch_cnt):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        # anim.add(epoch+1, train_metrics + (test_acc,))
        print("epoch %d, loss %f, train_accuracy %f, test_accuracy %f"
        %(epoch+1,train_metrics[0], train_metrics[1], test_acc))
        train_loss_lst.append(train_metrics[0])
        train_accuracy_lst.append(train_metrics[1])
        test_accuracy_lst.append(test_acc)
    return train_loss_lst,train_accuracy_lst,test_accuracy_lst


if __name__=='__main__':
    ip_cnt,op_cnt,hid1_cnt,hid2_cnt=28*28,10,256,256

    p1,p2=0.2,0.5

    epoch_cnt,lr,batch_size=10,0.5,256

    net=Net(ip_cnt,op_cnt,hid1_cnt,hid2_cnt,p1,p2)
    loss=nn.CrossEntropyLoss(reduction='none')
    trainer=tf.optim.SGD(net.parameters(),lr)

    train_iter,test_iter=load_data(batch_size)

    train_loss_lst,train_accuracy_lst,test_accuracy_lst=train(net,train_iter,test_iter,loss,epoch_cnt,trainer)

    plt.figure()
    plt.plot(list(range(1,epoch_cnt+1)),train_loss_lst,label='train loss')
    plt.plot(list(range(1,epoch_cnt+1)),train_accuracy_lst,label='train accuracy')
    plt.plot(list(range(1,epoch_cnt+1)),test_accuracy_lst,label='test accuracy')
    plt.legend()
    plt.show()