import torch as tf
from torch import nn


def corr2d(X,K):
    """二维互相关运算"""
    h,w=K.shape
    Y=tf.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y

class Conv2D(nn.Module):
    """卷积层，对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。"""
    def __init__(self,kernel_size):
        super().__init__()
        self.weight=nn.Parameter(tf.rand(kernel_size))
        self.bias=nn.Parameter(tf.zeros(1))
    def forward(self,X):
        return corr2d(X,self.weight)+self.bias

def train_pred_1(X,Y):
    """用nn自带卷积层实现"""
    # 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
    conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

    # 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
    # 其中批量大小和通道数都为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2  # 学习率
    trainer=tf.optim.SGD(conv2d.parameters(),lr)

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        trainer.step()
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')

    print("K:\n",conv2d.weight.data)


def train_pred_2(X,Y):
    """用自定义的卷积层实现"""
    conv2d = Conv2D((1,2))

    lr = 3e-2  # 学习率

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')

    print("K:\n",conv2d.weight.data)

if __name__=='__main__':
    X = tf.ones((6, 8))
    X[:, 2:6] = 0
    K0 = tf.tensor([[1.0, -1.0]])#初始卷积核
    Y=corr2d(X,K0)
    print("初始：")
    print("X:\n",X,"\n","Y:\n",Y,"\n","K0:\n",K0)
    print("---------------------------------------------")

    # train_pred_1(X,Y)
    train_pred_2(X,Y)