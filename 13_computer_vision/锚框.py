import torchvision as tcv
import matplotlib.pyplot as plt
import torch as tf


img=plt.imread("kano_2.jpg")

def get_rec(x,y,w,h,col,type):
    """用x,y,w,h坐标画矩形"""
    if type=='hw':
        return plt.Rectangle((x,y),w,h,fill=False,edgecolor=col,linewidth=2)
    elif type=='xy':
        return plt.Rectangle((x,y),w-x,h-y,fill=False,edgecolor=col,linewidth=2)

def multibox_prior(data, size, r):
    """
    生成以每个像素为中心具有不同形状的锚框
    输入：一个batchsize的图像数据、缩放比、高宽比
    """
    in_h, in_w=data.shape[-2:]
    device, size_cnt, r_cnt=data.device,len(size),len(r)
    boxes_pre_pix=size_cnt+r_cnt-1 #每个像素生成锚框数量
    size_tensor=tf.tensor(size,device=device)
    r_tensor=tf.tensor(r,device=device)

    """
    为了将锚点移动到像素的中心, 需要设置偏移量
    因为一个像素的高为1且宽为1, 我们选择偏移我们的中心0.5
    """
    d_h=1.0/in_h; d_w=1.0/in_w

    """生成锚框的所有中心点"""
    center_h=(tf.arange(in_h,device=device)+0.5)*d_h
    center_w=(tf.arange(in_w,device=device)+0.5)*d_w
    shift_y, shift_x=tf.meshgrid(center_h,center_w,indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    """对每个中心点生成一定数量锚框的高宽"""
    w=tf.cat((
            size_tensor*tf.sqrt(r_tensor[0]),
            size[0]*tf.sqrt(r_tensor[1:])
        )
    )* in_h / in_w #处理矩形输入
    h=tf.cat((
            size_tensor/tf.sqrt(r_tensor[0]),
            size[0]/tf.sqrt(r_tensor[1:])
        )
    )
    """除以2来获得半高和半宽"""
    anchor_manipulations = tf.stack((-w, -h, w, h)).T.repeat(
                                        in_h * in_w, 1) / 2
    print(anchor_manipulations.shape)
    # 每个中心点都将有“boxes_per_pix”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pix”次
    out_grid = tf.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_pre_pix, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

def show_bboxes(axes,bboxes,labels=None,colors=None):
    """显示所有边界框"""
    def get_lst(obj, default_values=None):
        """将label、colors转为list"""
        if obj is None:
            obj=default_values
        elif not isinstance(obj,(list,tuple)):
            obj=[obj]
        return obj
    labels=get_lst(labels); colors=get_lst(colors,['b', 'g', 'r', 'm', 'c'])

    for i,bbox in enumerate(bboxes):
        col=colors[i%len(colors)]
        rect=get_rec(*bbox.detach().numpy(),col,type='xy')
        axes.add_patch(rect)

        if labels and len(labels)>i:
            text_col='k' if col=='w' else 'w'
            axes.text(
                rect.xy[0],rect.xy[1],labels[i],
                va='center',ha='center',fontsize=9,color=text_col,
                bbox=dict(facecolor=col,lw=0)
            )


def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中两两成对的交并比"""
    get_S = lambda boxes: ((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))
    # boxes1,boxes2,S1,S2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # S1：(boxes1的数量,),
    # S2：(boxes2的数量,)
    S1=get_S(boxes1); S2=get_S(boxes2)
    """
    这里运用的广播机制
    boxes1.shape : [anchors_num, 4]
    boxes2.shape : [classes_num, 4]
    boxes1[:, None, :2].shape : [anchors_num, 1, 2]
    tf.max(boxes1[:, None, :2], boxes2[:, :2]).shape 
    = [anchors_num, classes_num, 2]
    通过广播机制能够将每个锚框与所有的真是边框进行计算, 也就是一个锚框与classes_num种      
    真实边框进行计算。
    """
    inter_upperlefts = tf.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = tf.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)#clamp将小于0的值变成0
    S_in = inters[:,:,0]*inters[:,:,1]
    return S_in/((S1[:,None]+S2)-S_in) 

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框:
    对于某个真实框：只取iou高于阈值的锚框，若所有锚框iou低于阈值，则取一个最佳锚框
    """
    anchors_cnt, gt_boxes_cnt=anchors.shape[0],ground_truth.shape[0]
    iou_sheet=box_iou(anchors,ground_truth)

    anchors_bbox_map=tf.full((anchors_cnt,),-1,dtype=tf.long,device=device)
    
    """根据阈值，决定是否分配真实边界框"""
    max_ious, indices=tf.max(iou_sheet,dim=1)#indice是最大值的索引
    anc_i=tf.nonzero(max_ious>=iou_threshold).reshape(-1)#取出满足iou阈值的下标
    box_j=indices[max_ious>=iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    # print(iou_sheet)
    # print(anc_i,end=' '); print(anchors_bbox_map,end=' ');print(box_j)

    col_discard = tf.full((anchors_cnt,), -1)#用于丢弃iou_sheet的行列
    row_discard = tf.full((gt_boxes_cnt,), -1)  

    for _ in range(gt_boxes_cnt):
        max_idx=tf.argmax(iou_sheet)
        box_idx=(max_idx % gt_boxes_cnt).long()#真实边框下标
        anc_idx=(max_idx / gt_boxes_cnt).long()#锚框下标
        anchors_bbox_map[anc_idx]=box_idx#将相应真实边框分配给锚框

        iou_sheet[:, box_idx] = col_discard#丢弃
        iou_sheet[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
    对锚框偏移量(中心坐标相对位置、框相对大小)的转换
    """
    def corner_to_center(a):
        b=tf.zeros(a.shape)
        b[:,0],b[:,1],b[:,2],b[:,3]=[
            (a[:,0]+a[:,2])/2,
            (a[:,1]+a[:,3])/2,
            a[:,2]-a[:,0],
            a[:,3]-a[:,1]
        ]
        return b
    anchors=corner_to_center(anchors)
    assigned_bb=corner_to_center(assigned_bb)
    offset_xy = 10 * (assigned_bb[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    offset_wh = 5 * tf.log(eps + assigned_bb[:, 2:] / anchors[:, 2:])
    offset = tf.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
            #label[:,0]是背景
            #本函数将背景类别的索引设置为零，然后将新类别的整数索引递增一。
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = tf.zeros(num_anchors, dtype=tf.long,
                                   device=device)
        assigned_bb = tf.zeros((num_anchors, 4), dtype=tf.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，我们标记其为背景（值为零）
        indices_true = tf.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = tf.stack(batch_offset)
    bbox_mask = tf.stack(batch_mask)
    class_labels = tf.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    def corner_to_center(a):
        b=tf.zeros(a.shape)
        b[:,0],b[:,1],b[:,2],b[:,3]=[
            (a[:,0]+a[:,2])/2,
            (a[:,1]+a[:,3])/2,
            a[:,2]-a[:,0],
            a[:,3]-a[:,1]
        ]
        return b
    def center_to_corner(a):
        b=tf.zeros(a.shape)
        b[:,0],b[:,1],b[:,2],b[:,3]=[
            a[:,0]-a[:,2]/2,
            a[:,1]-a[:,3]/2,
            a[:,0]+a[:,2]/2,
            a[:,1]+a[:,3]/2
        ]
        return b
    anc = corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = tf.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = tf.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B=tf.argsort(scores,dim=-1,descending=True)#返回排序后的索引值
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = tf.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return tf.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = tf.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = tf.arange(num_anchors, dtype=tf.long, device=device)
        combined = tf.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = tf.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = tf.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return tf.stack(out)


if __name__=='__main__':
    # fig=plt.imshow(img)
    # fig.axes.add_patch(
    #     get_rec(570,280,200,580,'green','hw')
    # )
    # fig.axes.add_patch(
    #     get_rec(820,400,300,450,'b','hw')
    # )
    # plt.show()

    """测试生成锚框"""
    # X=tf.rand(size=(1,3,720,480))
    # Y=multibox_prior(X,[0.75,0.5,0.25],[1,2,0.5])
    # print(Y.shape)

    """显示锚框"""
    # h,w=img.shape[:2]
    # X=tf.rand(size=(1,3,h,w))
    # boxes=multibox_prior(X,[0.5,0.35,0.25],[1,2,0.5])
    # boxes=boxes.reshape((h,w,5,4))

    # fig=plt.imshow(img)
    # show_bboxes(
    #     fig.axes, bboxes=boxes[550,680,:,:]*tf.tensor((w,h,w,h)),
    #     labels=['s=0.5, r=1', 's=0.35, r=1', 's=0.25, r=1', 's=0.5, r=2','s=0.5, r=0.5']
    # )
    # plt.show()

    """分配锚框"""
    # h,w=img.shape[:2]
    # ground_truth = tf.tensor(
    #     [[0, 0.35, 0.3, 0.47, 0.97],
    #     [1, 0.5, 0.4, 0.69, 0.95]]
    # )#每行第一个元素是类别下标、其余四个是坐上右下角xy坐标(01之间)
    # anchors = tf.tensor([[0.3, 0.1, 0.5, 0.8], [0.45, 0.2, 0.7, 0.6],
    #                     [0.63, 0.05, 0.88, 0.98], [0.46, 0.45, 0.6, 0.8],
    #                     [0.57, 0.3, 0.92, 0.9]])
    
    # fig=plt.imshow(img)
    # show_bboxes(fig.axes,ground_truth[:,1:]*tf.tensor((w,h,w,h)),['kano','deer'],'k')
    # show_bboxes(fig.axes,anchors*tf.tensor((w,h,w,h)),['0', '1', '2', '3', '4'])

    # mp=assign_anchor_to_bbox(ground_truth[:, 1:], anchors, anchors.device)
    # labels=multibox_target(anchors.unsqueeze(dim=0),ground_truth.unsqueeze(dim=0))
    
    # print(labels[2])
    # plt.show()

    """使用非极大值抑制来预测边界框"""
    h,w=img.shape[:2]
    ground_truth = tf.tensor(
        [[0, 0.35, 0.3, 0.47, 0.97],
        [1, 0.5, 0.4, 0.69, 0.95]]
    )#每行第一个元素是类别下标、其余四个是坐上右下角xy坐标(01之间)
    anchors = tf.tensor([[0.3, 0.1, 0.5, 0.8], [0.45, 0.2, 0.7, 0.6],
                        [0.63, 0.05, 0.88, 0.98], [0.46, 0.45, 0.6, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])
    offset_preds=tf.tensor([0 for i in range(anchors.numel())])
    iou_lst=box_iou(ground_truth[:,1:],anchors)
    cls_probs=tf.cat((tf.tensor([[0,1,1,0,1]]),iou_lst),dim=0)
    L=['background','kano','deer']
    lab=["%s=%.1f"%(L[tf.argmax(cls_probs[:,i])],tf.max(cls_probs[:,i])) for i in range(cls_probs.shape[1])]

    plt.figure()
    fig=plt.imshow(img)
    show_bboxes(fig.axes,anchors*tf.tensor((w,h,w,h)),lab)

    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.01)
    print(output)
    plt.figure()
    fig_2=plt.imshow(img)
    for i in output[0].detach().numpy():
        if i[0] == -1:
            continue
        label = ('kano=', 'deer=')[int(i[0])] + str(i[1])
        show_bboxes(fig_2.axes, [tf.tensor(i[2:]) * tf.tensor((w,h,w,h))], label)

    plt.show()