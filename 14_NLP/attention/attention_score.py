import sys
sys.path.append("../")

import math
import torch
from torch import nn
import torch.nn.functional as F
from attention.visualization import show_heatmaps

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange(
        (maxlen), dtype=torch.float32,
        device=X.device                    
    )[None, :] >= valid_len[:, None]#; print(mask,'\n',valid_len[:, None].shape)
    X[mask] = value
    return X


def mask_softmax(X, valid_lens = None):
    """通过在最后一个轴上mask元素来执行 softmax 操作"""
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape

        if valid_lens.dim() == 1:
            #valid_len中每个元素应用于一个batch
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            #valid_len中每个元素应用于一个batch中的一个序列
            valid_lens = valid_lens.reshape(-1)
        
        X = sequence_mask(
            X.reshape(-1,shape[-1]), valid_lens, -1e6
        )

        return (F.softmax(X.reshape(-1,shape[-1]),dim=-1)).reshape(shape)
    

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self,
            key_size:int,
            query_size:int,
            num_hiddens:int,
            dropout,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
        queries,
        keys,
        values,
        valid_lens,
    ):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1) 

        #(batch_size，查询的个数，“键－值”对的个数，num_hidden)
        features = torch.tanh(features)

        #(batch_size，查询的个数，“键－值”对的个数)
        score = self.W_v(features).squeeze(-1)

        self.attention_weight = mask_softmax(score, valid_lens)
        return torch.bmm(self.dropout(self.attention_weight),values)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, valid_lens=None):
        assert Q.shape[-1] == K.shape[-1]# shape = (batch,查询或键值对个数,维数)
        d = Q.shape[-1]
        scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = mask_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), V)
        


def show_attention_weight():
    Q, K, = torch.normal(0,1,(4,1,20)), torch.ones(4,10,2)
    V = torch.arange(40,dtype=torch.float32).reshape(1,10,4).repeat(4,1,1)
    valid_lens = torch.tensor([3,4,5,6])

    Attention_Block = AdditiveAttention(
        key_size=2,
        query_size=20,
        num_hiddens=8,
        dropout=0.1
    )
    Attention_Block.eval()
    print(Attention_Block(Q, K, V, valid_lens))
    print(Attention_Block.attention_weight.shape)
    show_heatmaps(Attention_Block.attention_weight.reshape((1,1,4,10)), 
                  xlabel='Keys', ylabel='Queries')

if __name__ == "__main__":
    # print(mask_softmax(torch.rand(2,2,4), torch.tensor([[1,3],[2,4]])))

    # show_attention_weight()
    a = torch.rand(2,3,4)
    print(a.transpose(0,2).shape)