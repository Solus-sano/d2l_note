import torch as tf
import os
from torch import nn
import torchvision as tcv
from torch.nn import functional as F
import matplotlib.pyplot as plt
from FAST_VOC_DATA import load_voc_data,read_voc_img,show_img
from Train_tool import train,accuracy
device='cuda' if tf.cuda.is_available() else 'cpu'


class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.ReLU(),

            nn.Conv2d(out_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.ReLU(),
        )
    
    def forward(self,X):
        return self.layer(X)

class Down_sample(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
    def forward(self,X):
        return self.layer(X)

class Up_sample(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
        self.c=Conv_Block(channel,channel//2)

    def forward(self,X,feature_map):
        op=F.interpolate(X,scale_factor=2,mode='nearest')
        op=self.layer(op)
        return self.c(tf.concat((op,feature_map),dim=1))
        

class UNET_net(nn.Module):
    def __init__(self):
        super().__init__()
        """下采样"""
        self.C1=Conv_Block(3,64)
        self.D1=Down_sample(64); self.C2=Conv_Block(64,128)
        self.D2=Down_sample(128); self.C3=Conv_Block(128,256)
        self.D3=Down_sample(256); self.C4=Conv_Block(256,512)
        self.D4=Down_sample(512); self.C5=Conv_Block(512,1024)
        """上采样"""
        self.U1=Up_sample(1024)
        self.U2=Up_sample(512)
        self.U3=Up_sample(256)
        self.U4=Up_sample(128)

        self.final_conv=nn.Conv2d(64,3,3,1,1)
    def forward(self,X):
        R1=self.C1(X)
        R2=self.C2(self.D1(R1))
        R3=self.C3(self.D2(R2))
        R4=self.C4(self.D3(R3))
        R5=self.C5(self.D4(R4))

        O1=self.U1(R5,R4)
        O2=self.U2(O1,R3)
        O3=self.U3(O2,R2)
        O4=self.U4(O3,R1)

        return self.final_conv(O4)

# def loss(Y_hat,Y):
#     return F.cross_entropy(Y_hat,Y,reduction='none').mean(1).mean(1)

if __name__ == '__main__':
    """读取VOC数据"""
    batchsize, crop_size=6,(256,320)
    train_iter,val_iter=load_voc_data(batchsize,crop_size,is_normal=False)

    unet_net=UNET_net()


    num_epochs, lr, wd = 10, 0.001, 1e-3
    loss_f=nn.CrossEntropyLoss()
    updater=tf.optim.Adam(unet_net.parameters(),lr,weight_decay=wd)

    # unet_net.load_state_dict(tf.load('fcn.pt'))

    unet_net.to(device)
    if not os.path.exists("unet_result_img"):
        os.mkdir("unet_result_img")
    for epoch in range(1,num_epochs+1):
        for i,(X,Y) in enumerate(train_iter):
            X=X.to(device); Y=Y.to(device)
            Y_hat=unet_net(X)
            l=loss_f(Y_hat,Y)

            updater.zero_grad()
            l.backward()
            updater.step()
            print("epoch %d batch %d loss: %f"%(epoch,i+1,float(l.sum())))

            if i%10==0:
                # seg_img,seg_img0=label2img(Y_hat.argmax(1)[0]),label2img(Y[0])
                # seg_img,seg_img0=seg_img.float()/255,seg_img0.float()/255
                op_img=tf.cat((X[0],Y_hat[0],Y[0]),dim=1)
                tcv.utils.save_image(op_img,f"unet_result_img/epoch_{epoch}_batch_{i}.jpg")
                # plt.imsave(f"unet_result_img/epoch_{epoch}_batch_{i}.jpg",op_img.cpu().detach().numpy())
        tf.save(unet_net.state_dict(),f'unet_epoch_{epoch}.pt')
    # train(unet_net,train_iter,val_iter,loss,num_epochs,updater)
    tf.save(unet_net.state_dict(),'unet.pt')
    plt.show()
    