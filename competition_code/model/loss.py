import torch
import torch.nn as nn
import torch.nn.functional as F

# add weights later
def BCE_loss(output, target):
    loss = nn.BCELoss()
    return loss(output, target)
