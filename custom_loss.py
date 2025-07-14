import torch
from torch import nn
import torch.nn.functional as F



class SigmoidLoss(nn.Module):
    
    def forward(self, p_loss, n_loss):
        p_loss = - F.logsigmoid(p_loss).mean()
        n_loss = - F.logsigmoid(-n_loss).mean()
        
        return (p_loss + n_loss) / 2, p_loss, n_loss 
