import torch as tf
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt

class Accumulator:
    """累加器"""
    def __init__(self,n):
        self.data=[0.0 for i in range(n)]

    def add(self,*args):#累加
        self.data=[a+float(b) for a,b in zip(self.data,args)]

    def reset(self):#归零
        self.data=[0.0 for i in range(len(self.data))]

    def __getitem__(self,idx):
        return self.data[idx]


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

def accuracy(y_hat,y):
    """计算一个batch的预测正确的样本数"""
    if y_hat.shape[0]>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)#取概率最大的下标作为预测类别
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())#预测正确的样本数

def init_weight(m):
    """初始化网络权重"""
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def evaluate_accuracy_gpu(net, data_iter,device=None):
    """评估模型在指定数据集上的精度"""
    if isinstance(net,tf.nn.Module):
        net.eval()#将模型设置为评估模式(即不计算梯度)
        if not device:
            device = next(iter(net.parameters())).device
    m = Accumulator(2)
    with tf.no_grad():
        for X,y in data_iter:
            X=X.to(device)
            y=y.to(device)
            m.add(accuracy(net(X),y),y.numel())
    return m[0]/m[1]

def train_gpu(net, train_iter, test_iter, loss, epoch_cnt, updater,device=None):
    """总训练函数"""
    loss_lst,train_accuracy_lst,test_accracy_lst=[],[],[]
    def train_epoch(net,train_iter,loss,updater,device):
        """训练模型时的单个迭代周期"""
        if isinstance(net,tf.nn.Module):
            net.train()#将模型设置为训练模式（可计算梯度）
        m = Accumulator(3)#累加器，（损失总和，预测正确样本数，样本总数）

        for X,y in train_iter:
            updater.zero_grad()
            X,y=X.to(device),y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)

            
            l.backward()
            updater.step()
            with tf.no_grad():
                m.add(float(l.sum()),accuracy(y_hat,y),y.numel())
        return m[0]/m[2] , m[1]/m[2]#loss,accuracy

    if not device:
        device = next(iter(net.parameters())).device

    net.apply(init_weight)
    print('train on ',device)
    net.to(device)
    for epoch in range(epoch_cnt):
        train_metrics = train_epoch(net, train_iter, loss, updater,device)
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # anim.add(epoch+1, train_metrics + (test_acc,))
        print("epoch %d, loss %f, train_accuracy %f, test_accuracy %f"
        %(epoch+1,train_metrics[0], train_metrics[1], test_acc))
        loss_lst.append(train_metrics[0])
        train_accuracy_lst.append(train_metrics[1])
        test_accracy_lst.append(test_acc)
    plt.figure()
    plt.plot(list(range(1,epoch_cnt+1)),train_accuracy_lst,label='train accuracy')
    plt.plot(list(range(1,epoch_cnt+1)),test_accracy_lst,label='test accuracy')
    plt.title('accuracy')
    plt.figure()
    plt.plot(list(range(1,epoch_cnt+1)),loss_lst,label='loss')
    plt.title('loss')
    # train_loss, train_acc = train_metrics


def predict(net, test_iter, cnt=6):
    """测试"""

    def labels_to_txt(labels):
        """
        返回数据集对应的文本标签
        """
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]

    def show_image(imgs,row_cnt,col_cnt,titles=None,scale=1.5):
        """
        绘制图像
        """
        fig_size=(col_cnt*scale,row_cnt*scale)
        _,axis = plt.subplots(row_cnt,col_cnt,figsize=fig_size)
        axis = axis.flatten()
        for i,(ax,img) in enumerate(zip(axis,imgs)):
            img=img[0]
            if tf.is_tensor(img):#图片张量
                ax.imshow(img.numpy())
            else:#PIL图片
                ax.imshow(img)
            if titles:
                ax.set_title(titles[i])
        return axis

    X,y=next(iter(test_iter))
    X, y = X.to(device), y.to(device)
    trues = labels_to_txt(y)
    preds = labels_to_txt(net(X).argmax(axis=1))
    titles=[t + '\n' + p for t,p in zip(trues,preds)]
    show_image(X[0:cnt].cpu(), 1, cnt, titles[0:cnt])



class Residual(nn.Module):
    """残差块"""
    def __init__(self,ip_channels,op_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(ip_channels,op_channels,
                            kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(op_channels,op_channels,
                            kernel_size=3,padding=1)

        if use_1x1conv:
            self.conv3=nn.Conv2d(ip_channels,op_channels,
                                kernel_size=1,stride=strides)
        else: self.conv3=None

        self.bn1=nn.BatchNorm2d(op_channels)
        self.bn2=nn.BatchNorm2d(op_channels)

    def forward(self,X):
        net1=nn.Sequential(self.conv1,self.bn1,nn.ReLU(),self.conv2,self.bn2)
        Y=net1(X)
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)


def resnet_block(ip_channels,op_channels,residuals_cnt,first_blk=False):
    blk=[]
    #第一个模块进行通道翻倍边长减半操作
    if not first_blk: blk.append(Residual(ip_channels,op_channels,use_1x1conv=True,strides=2))
    else: blk.append(Residual(op_channels,op_channels))
    for i in range(1,residuals_cnt):
        blk.append(Residual(op_channels,op_channels))
    return blk



#input shape: (1,1,96,96)
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#shape-> (1,64,24,24)
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_blk=True))#shape-> (1,64,24,24)
b3 = nn.Sequential(*resnet_block(64, 128, 2))#shape-> (1,128,12,12)
b4 = nn.Sequential(*resnet_block(128, 256, 2))#shape-> (1,256,6,6)
b5 = nn.Sequential(*resnet_block(256, 512, 2))#shape-> (1,512,3,3)

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),#shape-> (1,512,1,1)
                    nn.Flatten(),#shape-> (1,512)
                    nn.Linear(512, 10))#shape-> (1,10)



if __name__=='__main__':
    device="cuda" if tf.cuda.is_available() else "cpu"
    print(f"using {device} device")

    batch_size=256
    epoch_cnt=10
    lr=0.05
    train_iter, test_iter=load_data(batch_size,resize=96)

    net.apply(init_weight)
    loss=nn.CrossEntropyLoss()
    updater=tf.optim.Adam(net.parameters(),lr=lr)

    for X,y in train_iter:
        print(X)
        print(net(X).shape)
        print(net(X).dtype)
        break
    # train_gpu(net,train_iter,test_iter,loss,epoch_cnt,updater,device)
    # predict(net, test_iter)
    # plt.show()