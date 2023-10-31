import torch as tf
from torch.utils import data
import torchvision as tcv
import matplotlib.pyplot as plt

all_image=tcv.datasets.CIFAR10(
    root=r"../data",
    download=True,
    train=True
)

def show_img():
    img_tmp_lst=[all_image[i][0] for i in range(4*8)]
    _,axis=plt.subplots(4,8)
    axis=axis.flatten()
    for i,(ax,img) in enumerate(zip(axis,img_tmp_lst)):
        ax.imshow(img)
    return axis

def load_data(is_train,trans,batch_size):
    dataset=tcv.datasets.CIFAR10(
        root=r"../data",
        download=True,
        train=True,
        transform=trans
    )
    data_loader=data.DataLoader(dataset,batch_size=batch_size,shuffle=is_train,num_workers=4)
    return data_loader

if __name__=='__main__':

    train_trans=[
        tcv.transforms.RandomHorizontalFlip(),
        tcv.transforms.ToTensor()
    ]
    train_trans=tcv.transforms.Compose(train_trans)
    test_trans=tcv.transforms.Compose([tcv.transforms.ToTensor()])

    show_img()
    plt.show()