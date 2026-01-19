import torch.nn as nn
import torch.nn.functional as F

orderings = [
    [0, 1, 3, 4, 5],
    [1, 2, 0, 4, 5],
    [2, 3, 1, 4, 5],
    [3, 0, 2, 4, 5],
    [4, 1, 3, 2, 0],
    [5, 1, 3, 0, 2],
]
rotations = [
    [0, 0, 0, 0, 0],
    [0, 0, 0,-1, 1],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 1,-1],
    [0, 1,-1, 2, 0], 
    [0,-1, 1, 0, 2]
]

def _take_right(face, rot):
    if rot == 0:
        return face[:, :, 0]         
    elif rot == 1:
        return face[:, 0, :].flip(1) 
    elif rot == 2:
        return face[:, :, -1].flip(1)
    elif rot == -1:
        return face[:, -1, :]        

def _take_left(face, rot):
    if rot == 0:
        return face[:, :, -1]        
    elif rot == 1:
        return face[:, -1, :].flip(1)
    elif rot == 2:
        return face[:, :, 0].flip(1) 
    elif rot == -1:
        return face[:, 0, :]        

def _take_top(face, rot):
    if rot == 0:
        return face[:, -1, :]              
    elif rot == 1:
        return face[:, :, 0]               
    elif rot == 2:
        return face[:, 0, :].flip(1)       
    elif rot == -1:
        return face[:, :, -1].flip(1)      

def _take_bottom(face, rot):
    if rot == 0:
        return face[:, 0, :]               
    elif rot == 1:
        return face[:, :, -1]              
    elif rot == 2:
        return face[:, -1, :].flip(1)      
    elif rot == -1:
        return face[:, :, 0].flip(1)       

def valid_pad_conv_fn(x, one_side_pad=False):
    if one_side_pad:
        x = x[:, :, :-1, :-1]
    assert x.ndim == 4 and x.shape[0] == 6
    _, C, H, W = x.shape
    y = x.new_empty(6, C, H+2, W+2)
    y[..., 1:-1, 1:-1] = x

    for i in range(6):
        r_idx, l_idx, t_idx, b_idx = orderings[i][1:5]
        r_rot, l_rot, t_rot, b_rot = rotations[i][1:5]

        r_edge = _take_right (x[r_idx], r_rot)
        l_edge = _take_left  (x[l_idx], l_rot)
        t_edge = _take_top   (x[t_idx], t_rot)
        b_edge = _take_bottom(x[b_idx], b_rot)

        y[i, :, 1:-1, 0   ] = l_edge
        y[i, :, 1:-1, -1  ] = r_edge
        y[i, :, 0,     1:-1] = t_edge
        y[i, :, -1,    1:-1] = b_edge

        y[i, :, 0,  0 ] = 0.5*(y[i, :, 0, 1]   + y[i, :, 1, 0])
        y[i, :, 0, -1 ] = 0.5*(y[i, :, 0, -2]  + y[i, :, 1, -1])
        y[i, :, -1, 0 ] = 0.5*(y[i, :, -2, 0]  + y[i, :, -1, 1])
        y[i, :, -1,-1 ] = 0.5*(y[i, :, -2, -1] + y[i, :, -1, -2])

    if one_side_pad:
        return y[:, :, 1:, 1:]

    return y


class PaddedConv2d(nn.Conv2d):
    def __init__(self, *args, pad_fn=None, one_side_pad=False, **kwargs):
        kwargs = dict(kwargs)
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.pad_fn = pad_fn
        self.one_side_pad = one_side_pad

    def forward(self, x):
        x = self.pad_fn(x, one_side_pad=self.one_side_pad)
        return F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=0,
            dilation=self.dilation, groups=self.groups
        )

    @classmethod
    def from_existing(cls, conv: nn.Conv2d, pad_fn, one_side_pad=False):
        new = cls(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=0, dilation=conv.dilation,
            groups=conv.groups, bias=(conv.bias is not None),
            padding_mode="zeros", pad_fn=pad_fn, one_side_pad=one_side_pad
        )
        new.weight = conv.weight
        if conv.bias is not None:
            new.bias = conv.bias
        return new


