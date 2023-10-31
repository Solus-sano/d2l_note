import torch
from torch import nn
from torch.nn import functional as F
device="cuda" if torch.cuda.is_available() else "cpu"

def get_params(vovab_size, num_hidens):
    in_size = out_size = vovab_size

    def normal(shape):
        return nn.Parameter(torch.randn(size=shape, device=device) * 0.01)
    
    def three():
        W_x = normal((in_size, num_hidens))
        W_h = normal((num_hidens, num_hidens))
        b = normal((num_hidens, out_size))
        return (W_x, W_h, b)
    
    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()
    W_hq = normal((num_hidens, out_size))
    b_q = torch.zeros(out_size, device=device)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device = device):
    return ( torch.zeros((batch_size, num_hiddens), device=device) )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H,  = state
    outputs = []

    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tmp = torch.tanh((X @ W_xh) + (R @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tmp
        Y = H @ W_hq + b_q
        outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)

class GRU_model(nn.Module):
    def __init__(self,gru_layer,vocab_size):
        super().__init__()
        self.gru=gru_layer
        self.vocab_size=vocab_size
        self.num_hiddens=self.gru.hidden_size
        self.num_directions=1
        self.linear=nn.Linear(self.num_hiddens*self.num_directions,self.vocab_size)

    def forward(self,ip,state):
        X=F.one_hot(ip.T.long(),self.vocab_size)
        X=X.to(torch.float32)

        Y,state=self.gru(X,state)
        op=self.linear(Y.reshape((-1,Y.shape[-1])))
        return op,state

    def begin_state(self,device,batch_size=1):
        return torch.zeros((self.num_directions * self.gru.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)

