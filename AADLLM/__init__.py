import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):

    def __init__(self, in_dim, out_dim, n_layer=1, scale=True):
        super(FiLM, self).__init__()
        self.scale = scale
        if n_layer == 1:
            self.scaler = nn.Linear(in_dim, out_dim) if scale else None
            self.shifter = nn.Linear(in_dim, out_dim)
        elif n_layer == 2:
            self.scaler = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            ) if scale else None
            self.shifter = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )

    def forward(self, feat, cond):
        '''
        feat: (B, L, out_dim)
        cond: (B, in_dim)
        '''
        if self.scale:
            gamma = self.scaler(cond).unsqueeze(dim=1) # (B, 1, out_dim)
            beta = self.shifter(cond).unsqueeze(dim=1) # (B, 1, out_dim)
            cond_feat = feat * gamma + beta # time invariant

        else:
            beta = self.shifter(cond).unsqueeze(dim=1) # (B, 1, out_dim)
            cond_feat = feat + beta # time invariant
        
        return cond_feat
