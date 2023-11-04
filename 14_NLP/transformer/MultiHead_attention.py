import sys
sys.path.append("../")

import torch
import math
from torch import nn
from attention.attention_score import DotProductAttention

class MultiHead_Attention(nn.Module):
    def __init__(self,
            key_size,
            query_size,
            value_size,
            num_hiddens,
            num_heads: int,
            dropout: float,
            bias = False,
            **kwargs        
        ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)

        self.W_q = nn.Linear(query_size, num_hiddens * num_heads, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens * num_heads, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens * num_heads, bias=bias)
        self.W_o = nn.Linear(num_hiddens * num_heads, num_hiddens, bias=bias)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, valid_lens):
        """主函数"""
        Q = self.transpose_qkv(self.W_q(Q))
        K = self.transpose_qkv(self.W_k(K))
        V = self.transpose_qkv(self.W_v(V))
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens,
                repeats=self.num_heads,
                dim=0
            )
        
        output = self.attention(Q, K, V, valid_lens)
        # print(output.shape)
        return self.W_o(self.transpose_output(output))

    def transpose_qkv(self, X: torch.Tensor):
        """multihead并行计算预处理"""
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)

        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X: torch.Tensor):
        """逆转transpose_qkv"""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        
        return X.reshape(X.shape[0], X.shape[1], -1)

class PositionWiseFFN(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.dense1 = nn.Linear(in_ch, hid_ch)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hid_ch, out_ch)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        

class AddNorm(nn.Module):
    def __init__(self,nomalized_shape,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.LN = nn.LayerNorm(nomalized_shape)

    def forward(self, X, Y):
        return self.LN(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self,
            Q_size: int,
            K_size: int,
            V_size: int,
            num_hiddens: int,
            num_heads: int,
            dropout: int,
            ffn_in_ch: int,
            ffn_hid_ch: int,
            nomalized_shape: torch.Tensor,
            use_bias = False       
        ):
        super().__init__()

        self.attention = MultiHead_Attention(
            key_size=K_size,
            query_size=Q_size,
            value_size= V_size,
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias
        )
        self.addnorm1 = AddNorm(nomalized_shape,dropout)
        self.ffn = PositionWiseFFN(
            in_ch=ffn_in_ch,
            hid_ch=ffn_hid_ch,
            out_ch=num_hiddens
        )
        self.addnorm2 = AddNorm(nomalized_shape,dropout)

    def forward(self, X, valid_lens):
        X = self.addnorm1(X, self.attention(X,X,X,valid_lens))
        return self.addnorm2(X, self.ffn(X))

def test_MultiHead():
    attention = MultiHead_Attention(
        key_size=100,
        query_size=100,
        value_size=100,
        num_hiddens=200,
        num_heads=5,
        dropout=0.5
    ).eval()

    X = torch.ones((2,7,100))
    Y = torch.ones((2,13,100))
    valid_lens = torch.tensor([3,2])

    print(attention(X,Y,Y,valid_lens).shape)

def test_LN():
    ln = nn.LayerNorm(3)
    bn = nn.BatchNorm1d(3)
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    # 在训练模式下计算X的均值和方差
    print('layer norm:', ln(X).detach(),'\nbatch norm:', bn(X).detach())

def test_encoder():
    X = torch.ones([3,77,768])
    encoder = EncoderBlock(768,768,768,768,5,0.8,768,256,[77,768]).eval()
    valid_lens = torch.tensor([4,3,2])
    Y = encoder(X,valid_lens)
    print(Y.shape)

if __name__ == '__main__':
    test_encoder()
