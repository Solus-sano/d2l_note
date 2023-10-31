import torchvision as tcv
import matplotlib.pyplot as plt
from PIL import Image

def apply(img,aug,row_cnt=4,col_cnt=4,scale=1.5):
    
    Y=[aug(img) for i in range(row_cnt*col_cnt)]
    
    fig_size=(col_cnt*scale,row_cnt*scale)
    _,axis=plt.subplots(row_cnt,col_cnt,figsize=fig_size)
    axis=axis.flatten()
    for i,(ax,img) in enumerate(zip(axis,Y)):
        ax.imshow(img)
    return axis

if __name__=='__main__':
    img=Image.open(r"kano.jpg")
    # plt.imshow(img)
    trans_lst=[]

    """水平/垂直翻转"""
    trans_lst.append(tcv.transforms.RandomHorizontalFlip())
    trans_lst.append(tcv.transforms.RandomVerticalFlip())

    """随机裁剪"""
    trans_lst.append(tcv.transforms.RandomResizedCrop((200,200),#裁剪后resize的像素
                                                scale=(0.1,1),#裁剪比例范围
                                                ratio=(0.5,2)))#高宽比范围
    
    """随机更改图像亮度"""
    trans_lst.append(tcv.transforms.ColorJitter(
        brightness=0.5,#亮度（正负变化幅度）
        contrast=0.5,#对比度
        saturation=0.5,#饱和度
        hue=0.2#色调（正负变化幅度）
    ))

    """结合多种方法"""
    trans=tcv.transforms.Compose(trans_lst)
    apply(img,trans)
    plt.show()