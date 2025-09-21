import math
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from math import exp
from torch.autograd import Variable
from binary import *

def mse_loss_tumor(x, y,mask):
    x = [[1,2,3],[1,2,3],[1,2,3]]
    y = [[2,4,6],[2,4,6],[2,4,6]]
    mask = [[0,1,0],[0,1,0],[0,1,0]]
    epsilon = 1e-8
    A = x * mask
    B = y * mask
    return torch.mean((A-B)**2+epsilon)
   #return torch.mean((x*mask-y*mask)**2+epsilon)