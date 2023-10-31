import torch as tf
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import time

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
        # ax.axis.get_xaxis().set_visible(False)
        # ax.axis.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axis

def opt():
    trans = transforms.ToTensor()

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

    print("训练集大小：%d"%(len(mnist_train)))
    print("验证集大小：%d"%(len(mnist_test)))
    print("图片形状：")
    print(mnist_train[0][0].shape)
    """
    mnist_train[0]中为元组：（图片张量矩阵，类别编号）
    """

    # X , y=next(iter(data.DataLoader(mnist_train,batch_size=18)))
    # print(X.shape)
    # axis=show_image(X,2,9,titles=labels_to_txt(y))
    # plt.show()

    """
    读取小批量
    """
    batch_size = 256
    worker_cnt=4#读取数据时进程数
    train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=worker_cnt)

    time_begin=time.time()
    for X,y in train_iter:
        continue
    time_end=time.time()
    print("用时：%f"%(time_end-time_begin))
    #进程越多用时越多？？？

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
if __name__ =="__main__":
    # opt()
    train_iter,test_iter= load_data(32,resize=128)
    for X,y in train_iter:
        print(X.shape,y.shape)
        show_image(X,4,8,labels_to_txt(y))
        plt.show()
        break