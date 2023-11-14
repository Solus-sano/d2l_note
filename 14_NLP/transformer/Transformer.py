import math
import torch
from torch import nn
from MultiHead_attention import MultiHead_Attention, PositionWiseFFN, AddNorm

class PositionalEncoding(nn.Module):
    def __init__(self,
            num_hiddens: int,
            dropout: float,
            max_token_len = 1000    
        ) :
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_token_len, num_hiddens))

        tmp_col = torch.arange(max_token_len,dtype=torch.float32).reshape(-1,1)
        tmp_row = torch.pow(10000, torch.arange(0,num_hiddens,2, dtype=torch.float32) / num_hiddens)
        tmp_map = tmp_col / tmp_row

        self.P[:,:,::2] = torch.sin(tmp_map)
        self.P[:,:,1::2] = torch.cos(tmp_map)

    def forward(self, X: torch.Tensor):
        return self.dropout(X + self.P[:,:X.shape[1],:].to(X.device))
        

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
    

class DecoderBlock(nn.Module):
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
            block_idx: int,
            use_bias = False       
        ):
        super().__init__()

        self.block_idx = block_idx
        self.att_1 = MultiHead_Attention(
            key_size=K_size,
            query_size=Q_size,
            value_size=V_size,
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias
        )
        self.addnorm_1 = AddNorm(nomalized_shape, dropout)

        self.att_2 = MultiHead_Attention(
            key_size=K_size,
            query_size=Q_size,
            value_size=V_size,
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias
        )
        self.addnorm_2 = AddNorm(nomalized_shape, dropout)

        self.ffn = PositionWiseFFN(ffn_in_ch,ffn_hid_ch,num_hiddens)
        self.addnorm_3 = AddNorm(nomalized_shape, dropout)

    def forward(self, X: torch.Tensor, state):
        enc_op, enc_valid_lens = state[0], state[1]





class TransformerEncoder(nn.Module):
    def __init__(self,
            vocab_size: int,
            key_size: int,
            query_size: int,
            value_size: int,
            num_hiddens: int,
            num_heads: int,
            ffn_hid_ch: int,
            norm_shape: int,
            num_layers: int,
            dropout: float,
            use_bias = False,
            **kwargs         
        ):
        super().__init__(**kwargs)

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.ModuleList()

        for i in range(num_layers):
            self.blks.add_module(
                "block_" + str(i),
                EncoderBlock(
                    Q_size=query_size,
                    K_size=key_size,
                    V_size=value_size,
                    num_hiddens=num_hiddens,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_in_ch=num_hiddens,
                    ffn_hid_ch=ffn_hid_ch,
                    nomalized_shape=norm_shape,
                    use_bias=use_bias
                )
            )


    def forward(self, X: torch.Tensor, valid_lens: torch.Tensor):
        X = self.embedding(X)
        X = self.pos_encoding(X * math.sqrt(self.num_hiddens))#反转embedding中的范数归一化

        self.attention_weights = []
        for layer in self.blks:
            X = layer(X, valid_lens)
            self.attention_weights.append(layer.attention.attention.attention_weights)
        
        return X


def test_encoder():
    """测试encoder输出shape"""
    X = torch.ones([3,77,768])
    encoder = EncoderBlock(768,768,768,768,5,0.8,768,256,[77,768]).eval()
    valid_lens = torch.tensor([4,3,2])
    Y = encoder(X,valid_lens)
    print(Y.shape)

def test_transformer_encoder():
    trans_encoder = TransformerEncoder(
        vocab_size=6000,
        key_size=768,
        query_size=768,
        value_size=768,
        num_hiddens=768,
        num_heads=8,
        ffn_hid_ch=1024,
        norm_shape=[77,768],
        num_layers=12,
        dropout=0.8
    )
    trans_encoder.eval()
    valid_lens = torch.tensor([4,3,2])
    print(trans_encoder(torch.ones((3,77), dtype=torch.long), valid_lens).shape)

if __name__ == "__main__":
    test_transformer_encoder()