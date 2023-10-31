import os
import torch as tf
from torch.utils import data
import torchvision as tcv
import matplotlib.pyplot as plt



voc_dir="../data/VOCdevkit/VOC2012"


def show_img(img_lst,row_cnt=2,col_cnt=5,scale=1.5):
    """批量绘图"""
    fig_size=(col_cnt*scale,row_cnt*scale)
    _,axis=plt.subplots(row_cnt,col_cnt,figsize=fig_size)
    axis=axis.flatten()
    for i,(ax,img) in enumerate(zip(axis,img_lst)):
        ax.imshow(img)
    return axis

def read_voc_img(voc_dir,is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir,'ImageSets','Segmentation','train.txt' if is_train else 'val.txt')
    mode = tcv.io.image.ImageReadMode.RGB
    with open(txt_fname,'r') as f:
        img_name_lst=f.read().split()
    features,labels=[],[]
    for i,fname in enumerate(img_name_lst):
        features.append(tcv.io.read_image(os.path.join(voc_dir,'JPEGImages',f'{fname}.jpg')))
        labels.append(tcv.io.read_image(os.path.join(voc_dir,'SegmentationClass',f'{fname}.png'),mode))
    return features,labels

def rand_crop(feature,label,h,w):
    """随机裁剪(不伸缩)特征和标签图像"""
    rect=tcv.transforms.RandomCrop.get_params(feature,(h,w))#固定裁剪参数
    feature=tcv.transforms.functional.crop(feature, *rect)
    label=tcv.transforms.functional.crop(label, *rect)
    return feature,label

class VOC_Sec_Dataset(data.Dataset):
    def __init__(self,is_train,crop_size,voc_dir,is_normal=True):
        super().__init__()
        self.is_normal=is_normal
        self.trans=tcv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size=crop_size
        features,labels=read_voc_img(voc_dir,is_train)
        self.features=[self.normalize_img(fea) for fea in self.filter(features)]
        self.labels=[self.normalize_img(lab) for lab in self.filter(labels)]
        print("read %d examples"%(len(self.features)))

    def normalize_img(self,img):
        """归一化"""
        if self.is_normal:
            return self.trans(img.float()/255)
        else:
            return img.float()/255

    def filter(self,imgs):
        """去掉大小比裁剪框还小的图片"""
        return [img for img in imgs 
            if(img.shape[1]>=self.crop_size[0] and 
                img.shape[2]>=self.crop_size[1])
        ]
    
    def __getitem__(self, idx):
        """返回经过随机裁剪的第 idx 对数据"""
        fea,lab=rand_crop(self.features[idx],self.labels[idx],*self.crop_size)
        return fea,lab

    def __len__(self):
        return len(self.features)

def load_voc_data(batch_size,crop_size,is_normal=True):
    train_iter=data.DataLoader(
        VOC_Sec_Dataset(True,crop_size,voc_dir,is_normal),
        batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_iter=data.DataLoader(
        VOC_Sec_Dataset(False,crop_size,voc_dir,is_normal),
        batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_iter,val_iter

if __name__=='__main__':
    """数据集的前五张"""
    # train_features, train_labels = read_voc_img(voc_dir, True)
    # imgs = train_features[:5]+train_labels[:5]
    # imgs = [img.permute(1,2,0) for img in imgs]
    # show_img(imgs)
    # plt.show()

    """读取数据集"""
    crop_size=(320,480)
    train_dataset=VOC_Sec_Dataset(True,crop_size,voc_dir)
    val_dataset=VOC_Sec_Dataset(False,crop_size,voc_dir)

    batchsize=64
    train_iter=data.DataLoader(
        train_dataset,
        batchsize,
        shuffle=True,
        drop_last=True
    )
    for X,Y in train_iter:
        print(X.shape)
        print(Y.shape)
        break