# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        dp = F.pairwise_distance(anchor, positive)
        dn = F.pairwise_distance(anchor, negative)
        loss = F.relu(dp - dn + self.margin).mean()
        return loss

class AlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, emb_s, emb_p):
        return F.mse_loss(emb_s, emb_p)
