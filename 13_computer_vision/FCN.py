import torch as tf
from torch.nn import functional as F
import torchvision as tcv
import matplotlib.pyplot as plt
from VOC_DATA import load_voc_data,read_voc_img,VOC_COLORMAP,show_img
from Train_tool import train
device='cuda' if tf.cuda.is_available() else 'cpu'

pre_net = tcv.models.resnet18(pretrained=True)
net = tf.nn.Sequential(*list(pre_net.children())[:-2])

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (tf.arange(kernel_size).reshape(-1, 1),
          tf.arange(kernel_size).reshape(1, -1))
    filt = (1 - tf.abs(og[0] - center) / factor) * \
           (1 - tf.abs(og[1] - center) / factor)
    weight = tf.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

class FCN_net(tf.nn.Module):
    def __init__(self,class_cnt):
        super().__init__()
        pre_net = tcv.models.resnet18(pretrained=True)
        self.dowm_sampling_net=tf.nn.Sequential(*list(pre_net.children())[:-2])
        self.final_conv=tf.nn.Conv2d(512,class_cnt,1)
        self.trans_conv=tf.nn.ConvTranspose2d(class_cnt,class_cnt,64,padding=16,stride=32)
    
    def forward(self,X):
        op=self.dowm_sampling_net(X)
        op=self.final_conv(op)
        op=self.trans_conv(op)
        return op

def loss(Y_hat,Y):
    return F.cross_entropy(Y_hat,Y,reduction='none').mean(1).mean(1)


def predict(net,img):
    """预测"""
    trans=tcv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    X=trans(img.float()/255).unsqueeze(0)
    pred=net(X.to(device)).argmax(dim=1)
    return pred[0]

def label2img(pred):
    """将预测类别映射回它们在数据集中的标注颜色"""
    col_map=tf.tensor(VOC_COLORMAP,device=device)
    X=pred.long()
    return col_map[X,:]

if __name__=='__main__':
    """读取VOC数据"""
    batchsize, crop_size=32,(320,480)
    train_iter,val_iter=load_voc_data(batchsize,crop_size)


    X=tf.rand(size=(1,3,320,480))
    fcn_net=FCN_net(21)
    print(fcn_net(X).shape)
    # print(list(fcn_net.children())[-3:])
    """双线性插值转置卷积层实验"""
    # img=tcv.transforms.ToTensor()(plt.imread("kano.jpg"))
    # img=img.unsqueeze(0)
    # tmp_conv=tf.nn.ConvTranspose2d(3,3,3,padding=0,stride=2,bias=False)
    # tmp_conv.weight.data.copy_(bilinear_kernel(3,3,3))
    # img=tmp_conv(img)
    # img=img[0].permute(1,2,0)
    # plt.imshow(img.detach())
    # plt.show()

    """训练"""
    # fcn_net.trans_conv.weight.data.copy_(bilinear_kernel(21,21,64))


    # num_epochs, lr, wd = 10, 0.001, 1e-3

    # updater=tf.optim.SGD(fcn_net.parameters(),lr,weight_decay=wd)

    # # fcn_net.load_state_dict(tf.load('fcn.pt'))

    # fcn_net.to(device)
    # train(fcn_net,train_iter,val_iter,loss,num_epochs,updater)
    # tf.save(fcn_net.state_dict(),'fcn.pt')
    # plt.show()

    """预测"""
    # fcn_net=FCN_net(21)
    # fcn_net.load_state_dict(tf.load('fcn.pt'))
    # test_images, test_labels=read_voc_img(voc_dir="../data/VOCdevkit/VOC2012",is_train=False)
    # fcn_net.to(device)

    # n,img_lst=5,[]
    # for i in range(20,20+n):
    #     rect=(0,0,320,480)
    #     X=tcv.transforms.functional.crop(test_images[i],*rect)
    #     pred=label2img(predict(fcn_net,X))
    #     img_lst+=[
    #         X.permute(1,2,0),
    #         pred.cpu(),
    #         tcv.transforms.functional.crop(test_labels[i],*rect).permute(1,2,0)
    #     ]
    
    # show_img(img_lst[::3]+img_lst[1::3]+img_lst[2::3],3,n,2)

    # plt.show()

