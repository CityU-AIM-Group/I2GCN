import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math
import numpy as np 
from scipy.special import softmax


def A_normalize(
    matrix, 
    self_loop, 
    self_loop_flag=True):
    # normalization for each adjacency matrix [K, K]
    matrix = F.relu(matrix, inplace=False)
    if self_loop_flag:
        matrix = matrix + self_loop # self.diagonal_i
    with torch.no_grad():
        degree = torch.diag(torch.pow(torch.sum(matrix, 1), -0.5))

    return torch.mm(degree, torch.mm(matrix, degree))

def build_adjacent_Feature(
    x, 
    adj_ratio, 
    self_loop, 
    self_loop_flag=True):
    b, c, n = x.shape
    x_T = torch.transpose(x, 1, 2) # b*n*c2
    adjacency_matrix = torch.bmm(x_T, x) # b*n*n
    # threshold
    threshold_idx = int(adj_ratio * n * n + 0.5) 
    adj_list = []
    m1 = torch.ones(n, n)
    m0 = torch.zeros(n, n)
    if adjacency_matrix.is_cuda:
        m1 = m1.cuda()
        m0 = m0.cuda()
    for adj_matrix_sample in adjacency_matrix: # batch loop
        scores, _ = torch.topk(adj_matrix_sample.view(-1), k=threshold_idx, largest=True, sorted=True)
        threshold = scores[-1]
        adj_matrix_binary = torch.where(adj_matrix_sample >= threshold, m1, m0)
        adj_list.append(adj_matrix_binary)
    #
    adjacency_matrix = torch.stack(adj_list)
    # normalize adj (b*n*n)
    adjacency_matrix = torch.stack(
        [
            A_normalize(
                a, 
                self_loop=self_loop, 
                self_loop_flag=self_loop_flag
            ) for a in adjacency_matrix
        ]
    ) # B*Z*Z
    return adjacency_matrix

def instance_gassian(heatmap, pt, sigma):
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)] # 左上
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)] # 右下

    size = 6 * sigma + 1
    x = torch.arange(0, size, 1)
    y = torch.arange(0, size, 1)
    x0 = y0 = float(size//2)
    g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heatmap.shape[0]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heatmap.shape[1]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], heatmap.shape[0])
    img_y = max(0, ul[1]), min(br[1], heatmap.shape[1])
    heatmap[img_x[0]:img_x[1], img_y[0]:img_y[1]] += g[g_x[0]:g_x[1]].view(-1, 1)*g[g_y[0]:g_y[1]].view(1, -1)
    return heatmap
        
def build_adjacent_PriorImportance(
    x, 
    importances_batch, 
    sigma,
    self_loop, 
    self_loop_flag=True):
    # importances_batch: [B, 1, K]
    B, C, K = x.shape
    adjacency_matrix_list = []
    for b in range(B):
        adjacency_matrix_initial = torch.zeros(K, K)
        adjacency_matrix_sample = instance_gassian(
            heatmap=adjacency_matrix_initial, 
            pt=importances_batch[b][0],
            sigma=sigma
        )
        ##
        adjacency_matrix_list.append(adjacency_matrix_sample) # len: B
    #
    adjacency_matrix = torch.stack(adjacency_matrix_list, dim=0) # B*K*K
    if x.is_cuda:
        adjacency_matrix = adjacency_matrix.cuda() 
    # normalize adj (B*K*K)
    adjacency_matrix = torch.stack(
        [
            A_normalize(
                a,
                self_loop=self_loop, 
                self_loop_flag=self_loop_flag
            ) for a in adjacency_matrix
        ]
    ) # B*K*K
    return adjacency_matrix


class I2GCLayer(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel,  
        bias=False):
        # if res_connct is true, in_channel should be equal to out_channel
        super(I2GCLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = nn.Parameter(torch.FloatTensor(out_channel, in_channel), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_channel, 1), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, 0.01)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adjacency_matrix_list):
        # x [B, C, K]
        # adjacency_matrix [B, K, K]
        # AXW
        output_tmp1 = torch.stack([torch.mm(self.weight[:int(self.out_channel/2)], torch.mm(a, b)) for a, b in zip(x, adjacency_matrix_list[0])])
        output_tmp2 = torch.stack([torch.mm(self.weight[int(self.out_channel/2):], torch.mm(a, b)) for a, b in zip(x, adjacency_matrix_list[1])])
        output = torch.cat([output_tmp1, output_tmp2], dim=1)
        if self.bias is not None: # GCN bias
            output = output + self.bias

        output = F.relu(output, inplace=False)

        return output
