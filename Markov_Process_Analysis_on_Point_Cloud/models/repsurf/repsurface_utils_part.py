import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from collections import OrderedDict


from modules.pointnet2_utils import farthest_point_sample, index_points, query_knn_point, query_ball_point
from modules.polar_utils import xyz2sphere
from modules.recons_utils import cal_const, cal_normal, cal_center, check_nan_umb



#############################################################
# import dgl.function as dgl_fn
# import dgl
#############################################################


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


# def pc_normalize(pc):
#     l = pc.shape[0]
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc = pc / m
#     return pc


# def square_distance(src, dst):
#     """
#     Calculate Euclid distance between each two points.
#
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist


# def index_points(points, idx):
#     """
#
#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#     return new_points


# def mod_index(bse_xyz, mod_idx, xyz):
#     # bse_xyz in [B, N, D]
#     batch_indexes = torch.arange(mod_idx.shape[0]).unsqueeze(dim=-1).repeat(1, mod_idx.shape[1]).flatten()
#     modidx_indexes = mod_idx.flatten()
#     mask = torch.ones_like(bse_xyz)
#     mask[batch_indexes, modidx_indexes, :] = 0.0
#     expand_xyz = torch.zeros_like(bse_xyz)
#     expand_xyz[batch_indexes, modidx_indexes, :] = xyz.view(-1, xyz.shape[-1])
#     return bse_xyz * mask + expand_xyz
#
#
# def index_points2(points, idx, new_points):
#     """
#
#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     points[batch_indices, idx, :] = new_points  #######################
#     return points


# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#     for i in range(npoint):
#         centroids[:, i] = farthest
#
#         # a = xyz[batch_indices,:,:]       ###########################
#         # b = xyz[batch_indices,farthest,:]   #######################
#         # centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)   ################%%%%%%%%%%%%%%%%%%
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, C)  ######################################
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#     return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    # dist = torch.gather(sqrdists, -1, group_idx)   #################
    return group_idx  # , dist


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


##############################################################

# def conv_bn(inp, oup, kernel, stride=1, activation='relu'):
#     seq = nn.Sequential(
#         nn.Conv2d(inp, oup, kernel, stride),
#         nn.BatchNorm2d(oup)
#     )
#     if activation == 'relu':
#         seq.add_module('2', nn.ReLU())
#     return seq
#

# def square_distance(src, dst):
#     """
#     Calculate Euclid distance between each two points.
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist
#
#
# def knn_point(nsample, xyz, new_xyz):
#     """
#     Input:
#         nsample: max sample number in local region
#         xyz: all points, [B, N, C]
#         new_xyz: query points, [B, S, C]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     sqrdists = square_distance(new_xyz, xyz)
#     dist, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
#     return dist, group_idx
#
#
# def knn(x, k):
#     x = x.transpose(2, 1)  ################################
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx
#
#
# def pointsift_select(radius, xyz):
#     """
#     code by python matrix logic
#     :param radius:
#     :param xyz:
#     :return: idx
#     """
#     dev = xyz.device
#     B, N, _ = xyz.shape
#     judge_dist = radius ** 2
#     idx = torch.arange(N).repeat(8, 1).permute(1, 0).contiguous().repeat(B, 1, 1).to(dev)
#
#     distenceNN = square_distance(xyz, xyz)  ########################
#
#     for n in range(N):
#         distance = torch.ones(B, N, 8).to(dev) * 1e10
#         distance[:, n, :] = judge_dist
#         centroid = xyz[:, n, :].view(B, 1, 3).to(dev)
#         dist = torch.sum((xyz - centroid) ** 2, -1)  # shape: (B, N)
#
#         # subspace_idx = torch.sum((xyz - centroid + 1).int() * torch.tensor([4, 2, 1], dtype=torch.int, device=dev), -1)   ##################%%%%%%%%%%%%%%%%%%%%
#         subspace_idx = torch.sum((xyz > centroid).int() * torch.tensor([4, 2, 1], dtype=torch.int, device=dev),
#                                  -1)  #################################
#
#         for i in range(8):
#             mask = (subspace_idx == i) & (dist > 1e-10) & (dist < judge_dist)  # shape: (B, N)
#             distance[..., i][mask] = dist[mask]
#             c = torch.min(distance[..., i], dim=-1)[1]
#             idx[:, n, i] = torch.min(distance[..., i], dim=-1)[1]
#     return idx
#
#
# def pointsift_group(radius, xyz, points, use_xyz=True):
#     B, N, C = xyz.shape
#     assert C == 3
#     idx = pointsift_select(radius, xyz)  # B, N, 8
#
#     grouped_xyz = index_points(xyz, idx)  # B, N, 8, 3
#
#     # grouped_xyz -= xyz.view(B, N, 1, 3)   ######################%%%%%%%%%%%%%%%%%%%%%%%
#
#     xyz = xyz.view(B, N, 1, 3).repeat(1, 1, 8, 1)  #####################################
#     grouped_edge = torch.cat((grouped_xyz - xyz, xyz), dim=3).contiguous()  ##################################
#     # grouped_points = grouped_edge   ######################################
#
#     if points is not None:
#         grouped_points = index_points(points, idx)
#         if use_xyz:
#             grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
#     else:
#         # grouped_points = grouped_xyz  ################%%%%%%%%%%%%%%%%%%%%%%
#         grouped_points = grouped_xyz - xyz  ######################################
#
#     return grouped_xyz, grouped_points, grouped_edge, idx
#
#
# def conv1d(inplanes, outplanes, stride=1):
#     return nn.Sequential(
#         nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
#         nn.BatchNorm1d(outplanes),
#         nn.LeakyReLU(inplace=True, negative_slope=0.2)
#     )
#
#
# def convFuse(inplanes, outplanes, stride=1):
#     return nn.Sequential(
#         nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
#         nn.BatchNorm2d(outplanes))
#
#
# def fc(inplanes, outplanes):
#     return nn.Sequential(
#         nn.Linear(inplanes, outplanes, bias=False),
#         nn.BatchNorm1d(outplanes))


# class KeepHighResolutionModule(nn.Module):
#     def __init__(self, num_branches, blocks, num_block, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
#         super(KeepHighResolutionModule, self).__init__()
#
#         self.num_inchannels = num_inchannels
#         self.fuse_method = fuse_method
#         self.num_branches = num_branches
#         self.multi_scale_output = multi_scale_output
#
#         self.branches = self.make_branches(num_branches, blocks, num_block, num_channels)
#
#         self.fuse_layers = self.make_fuse()
#         self.relu = nn.ReLU(inplace=True)
#
#     def make_one_branch(self, branch_index, block, num_blocks, num_channels):
#         layers = []
#         layers.append(
#             block(self.num_inchannels[branch_index],
#                   num_channels[branch_index])
#         )
#
#         for i in range(1, num_blocks[branch_index]):
#             layers.append(
#                 block(self.num_inchannels[branch_index],
#                       num_channels[branch_index])
#             )
#
#         return nn.Sequential(*layers)
#
#     def make_branches(self, num_branches, block, num_blocks, num_channels):
#         branches = []
#
#         for i in range(num_branches):
#             branches.append(
#                 self.make_one_branch(i, block, num_blocks, num_channels)
#             )
#
#         return nn.ModuleList(branches)
#
#     def forward(self, x):
#         for i in range(self.num_branches):
#             x[i] = self.branches[i](x[i])
#
#         x_fuse = []
#
#
#         return x_fuse


# def make_fuse(npoint_list):
#     for i in range(len(npoint_list)):
#         branch1 = npoint_list[0]
#         branch2 = npoint_list[1]
#
#
# def random_sample(xyz, sample_num):
#     B, N, _ = xyz.size()
#     permutation = torch.randperm(N)
#     temp_sample = xyz[:, permutation]
#     sampled_xyz = temp_sample[:, :sample_num, :]
#
#     idx = permutation[:sample_num].unsqueeze(0).expand(B, sample_num)
#
#     return sampled_xyz, idx
#
#
# def sample_anchors(x, s):
#     idx = torch.randperm(x.size(3))[:s]
#     x = x[:, :, :, idx]
#
#     return x


# def make_head(pre_stage_channels):
#     head_channels = [32, 64, 128, 256]
#
#     incre_modules = []
#     for i, channels in enumerate(pre_stage_channels):
#         incre_module = SharedMLP(channels, head_channels[i] * 4, bn=True,
#                                  activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         incre_modules.append(incre_module)
#
#     incre_modules = nn.ModuleList(incre_modules)
#
#     downsamp_modules = []
#     for i in range(len(pre_stage_channels) - 1):
#         in_channel = head_channels[i + 1]
#         out_channel = head_channels[i + 1] * 2
#
#         downsamp_module = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         downsamp_modules.append(downsamp_module)
#
#     downsamp_modules = nn.ModuleList(downsamp_modules)
#
#     fuse_modules = []
#     for i in range(len(pre_stage_channels) - 1):
#         in_channel = head_channels[i] * 4
#         out_channel = head_channels[i + 1] * 4
#
#         fuse_module = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         fuse_modules.append(fuse_module)
#
#     fuse_modules = nn.ModuleList(fuse_modules)
#
#     return incre_modules, downsamp_modules, fuse_modules
#
#
# def make_stage(stage_channels):
#     stage_modules = []
#
#     for i in range(len(stage_channels)):
#         in_channel = stage_channels[i]
#         out_channel = stage_channels[i]
#
#         stage_module = BasicBlock(in_channel, out_channel)
#         stage_modules.append(stage_module)
#
#     stage_modules = nn.Sequential(stage_modules[0], stage_modules[1])
#
#     return stage_modules


# def convert_polar(neighbours, center):
#     neighbours = neighbours.permute(0, 2, 3, 1).contiguous()
#     center = center.permute(0, 2, 3, 1).contiguous()
#
#     rel_x = (neighbours - center)[:, :, :, 0]
#     rel_y = (neighbours - center)[:, :, :, 1]
#     rel_z = (neighbours - center)[:, :, :, 2]
#
#     r_xy = torch.sqrt(rel_x ** 2 + rel_y ** 2)
#     r_zx = torch.sqrt(rel_z ** 2 + rel_x ** 2)
#     r_yz = torch.sqrt(rel_y ** 2 + rel_y ** 2)
#
#     ### Z_axis
#     z_beta = torch.atan2(rel_z, r_xy).unsqueeze(-3).contiguous()
#     z_alpha = torch.atan2(rel_y, rel_x).unsqueeze(-3).contiguous()
#
#     ### Y_axis
#     y_beta = torch.atan2(rel_y, r_zx).unsqueeze(-3).contiguous()
#     y_alpha = torch.atan2(rel_x, rel_z).unsqueeze(-3).contiguous()
#
#     ### X_axis
#     x_beta = torch.atan2(rel_x, r_yz).unsqueeze(-3).contiguous()
#     x_alpha = torch.atan2(rel_z, rel_y).unsqueeze(-3).contiguous()
#
#     return x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta
#
#
# def Gaussian(features):
#     features = features.permute(0, 2, 3, 1)
#     B, N, K, C = features.shape
#
#     # mean = torch.mean(features, dim=2, keepdim=True)
#     std = torch.std((features).reshape(B, -1), dim=-1, unbiased=False, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
#
#     feature = (features) / (std + 1e-5)
#
#     return feature
#
#
# def conv3x3(in_channels, out_channels, kernel, stride=1):
#     return nn.Sequential(
#         # nn.BatchNorm2d(in_channels, momentum=0.99),
#         # nn.LeakyReLU(negative_slope=0.2),
#         nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=False),
#         # nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, bias=False)
#         nn.BatchNorm2d(out_channels, momentum=0.99),
#         nn.LeakyReLU(negative_slope=0.2)
#     )
#
#
# def conv1x1(in_channels, out_channels, stride=1):
#     return nn.Sequential(
#         # nn.BatchNorm2d(in_channels, momentum=0.99),
#         # nn.LeakyReLU(negative_slope=0.2),
#         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
#         nn.BatchNorm2d(out_channels, momentum=0.99),
#         nn.LeakyReLU(negative_slope=0.2)
#     )

#
# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, outplanes):
#         super(BasicBlock, self).__init__()
#
#         # self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1)
#         # self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-6, momentum=0.99)
#         # self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1)
#         # self.bn2 = nn.BatchNorm2d(inplanes, eps=1e-6, momentum=0.99)
#         self.conv1 = SharedMLP(inplanes, outplanes, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = SharedMLP(outplanes, outplanes, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = SharedMLP(outplanes, outplanes, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         # out = self.bn1(out)
#         # out = self.lrelu(out)
#
#         out = self.conv2(out)
#         # out = self.bn2(out)
#
#         out = self.conv3(out)
#
#         # out = self.lrelu(out + residual)
#
#         return out


# class Linear(nn.Module):
#     def __init__(self, in_channels, out_channels, bn=True, act=True):
#         super(Linear, self).__init__()
#
#         self.act_flag = act
#         self.bn_flag = bn
#
#         self.linear = nn.Linear(in_channels, out_channels)
#         self.norm1 = nn.LayerNorm(out_channels)
#         self.norm2 = nn.BatchNorm1d(out_channels)
#         self.act = nn.LeakyReLU(negative_slope=0.2)
#
#     def forward(self, input):
#
#         out = self.linear(input)
#
#         if self.bn_flag is True:
#             out = self.norm1(out)
#         else:
#             out = self.norm2(out.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
#
#         if self.act_flag is True:
#             out = self.act(out)
#
#         return out


# class SharedMLP(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size=1,
#             stride=1,
#             transpose=False,
#             padding_mode='zeros',
#             bn=False,
#             activation_fn=None
#     ):
#         super(SharedMLP, self).__init__()
#
#         conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d
#
#         self.conv = conv_fn(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding_mode=padding_mode
#         )
#         self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
#         # self.batch_norm = nn.BatchNorm2d(out_channels) if bn else None  #################
#         self.activation_fn = activation_fn
#
#     def forward(self, input):
#         r"""
#             Forward pass of the network
#
#             Parameters
#             ----------
#             input: torch.Tensor, shape (B, d_in, N, K)
#
#             Returns
#             -------
#             torch.Tensor, shape (B, d_out, N, K)
#         """
#         x = self.conv(input)
#         if self.batch_norm:
#             x = self.batch_norm(x)
#         if self.activation_fn:
#             x = self.activation_fn(x)
#
#         # if self.batch_norm:                  ##############
#         #     x = self.batch_norm(input)    ##############
#         # if self.activation_fn:                ##############
#         #     x = self.activation_fn(x)         ###########
#         # x = self.conv(x)   ##########################
#         return x
#
#
# class SharedMLP2(nn.Module):
#     def __init__(self, in_channels, out_channels, residual=False, first=False):
#         super(SharedMLP2, self).__init__()
#
#         # conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d
#         #
#         # self.conv = conv_fn(
#         #     in_channels,
#         #     out_channels,
#         #     kernel_size,
#         #     stride=stride,
#         #     padding_mode=padding_mode,
#         #     bias=bias
#         # )
#         # self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
#         # self.activation_fn = activation_fn
#         self.residual = residual
#         self.first = first
#
#         self.conv1 = conv3x3(in_channels, out_channels, (1, 1))
#         self.conv2 = conv3x3(in_channels, out_channels, (1, 1))
#         self.conv3 = conv3x3(in_channels, out_channels, (1, 1))
#
#         self.conv1x1 = conv1x1(out_channels * 3, out_channels)
#
#         self.conv1x1_residual = conv1x1(in_channels, out_channels)
#
#     def forward(self, xyz, new_xyz, points=None, FPS_idx=None):
#         r"""
#             Forward pass of the network
#
#             Parameters
#             ----------
#             input: torch.Tensor, shape (B, d_in, N, K)
#
#             Returns
#             -------
#             torch.Tensor, shape (B, d_out, N, K)
#         """
#
#         dist, idx = knn_point(49, xyz.permute(0, 2, 1), new_xyz.permute(0, 2, 1))  ### B N K
#
#         idx1 = idx[:, :, 0:16]
#         # idx2 = idx[:,:,0:25]
#         # idx3 = idx[:,:,0:49]
#
#         if points is None:
#             input1 = index_points(xyz.permute(0, 2, 1), idx1).permute(0, 3, 1, 2) - xyz.unsqueeze(-1)
#             # input2 = index_points(xyz.permute(0, 2, 1), idx2).permute(0, 3, 1, 2) - xyz.unsqueeze(-1)
#             # input3 = index_points(xyz.permute(0, 2, 1), idx3).permute(0, 3, 1, 2) - xyz.unsqueeze(-1)
#
#             x1 = self.conv1(input1)
#             # x2 = self.conv2(input2)
#             # x3 = self.conv3(input3)
#
#             x = torch.max(x1, 3)[0]
#             # x2 = torch.max(x2, 3)[0]
#             # x3 = torch.max(x3, 3)[0]
#
#             # x = torch.cat((x1, x2, x3), dim=1)
#             # x = self.conv1x1(x.unsqueeze(-1)).squeeze(-1)
#
#
#         else:
#             center = index_points(points.permute(0, 2, 1), FPS_idx).permute(0, 2, 1).unsqueeze(-1)
#
#             input1 = index_points(points.permute(0, 2, 1), idx1).permute(0, 3, 1, 2) - center
#             # input2 = index_points(points.permute(0, 2, 1), idx2).permute(0, 3, 1, 2) - center
#             # input3 = index_points(points.permute(0, 2, 1), idx3).permute(0, 3, 1, 2) - center
#
#             x1 = self.conv1(input1)
#             # x2 = self.conv2(input2)
#             # x3 = self.conv3(input3)
#
#             x = torch.max(x1, 3)[0]
#             # x2 = torch.max(x2, 3)[0]
#             # x3 = torch.max(x3, 3)[0]
#
#             # x = torch.cat((x1, x2, x3), dim=1)
#
#             # x = self.conv1x1(x.unsqueeze(-1)).squeeze(-1)
#
#         if FPS_idx is not None:
#             residual = index_points(points.permute(0, 2, 1), FPS_idx).permute(0, 2, 1)
#
#             if self.residual == True:
#                 residual = self.conv1x1_residual(residual.unsqueeze(-1)).squeeze(-1)
#
#             x = x + residual
#
#         if self.first == True:
#             residual = self.conv1x1_residual(xyz.unsqueeze(-1)).squeeze(-1)
#
#             x = x + residual
#
#         return x
#
#         # if points is None:
#         #     input = index_points(xyz.permute(0, 2, 1), idx).permute(0, 3, 1, 2)
#         #
#         #     x = self.conv(input)
#         #     if self.batch_norm:
#         #         x = self.batch_norm(x)
#         #     if self.activation_fn:
#         #         x = self.activation_fn(x)
#         #
#         #     return x
#         #
#         # else:
#         #     input = index_points(points.permute(0, 2, 1), idx).permute(0, 3, 1, 2)
#         #     # fps_input = index_points(input.permute(0,2,3,1), FPS_idx).permute(0,3,1,2)
#         #
#         #     x = self.conv(input)
#         #     if self.batch_norm:
#         #         x = self.batch_norm(x)
#         #     if self.activation_fn:
#         #         x = self.activation_fn(x)
#         #
#         #     return x
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(SpatialAttention, self).__init__()
#
#         # self.conv = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.conv = SharedMLP(in_c, out_c // 4, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         _, K, _, _ = x.size()
#         context = self.conv(x)
#         context = context.repeat(1, K // 2, 1, 1)
#
#         context = self.sigmoid(context)
#
#         x = x + torch.mul(context, x)
#
#         # avgout = torch.mean(x, dim=1, keepdim=True)
#         # maxout, _ = torch.max(x, dim=1, keepdim=True)
#         # out = torch.cat([avgout, maxout], dim=1)
#         # out = self.conv(out)
#         # out = self.sigmoid(out)
#         #
#         # out = torch.sum(out * x, dim=-1, keepdim=True)
#
#         return x
#
#
# class AttentivePooling(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction):
#         super(AttentivePooling, self).__init__()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         # self.avg_pool1 = nn.AdaptiveAvgPool1d(1)
#
#         self.score_fn = nn.Sequential(
#             nn.Linear(in_channels, in_channels, bias=False),
#             # nn.Conv2d(in_channels, reduction, kernel_size=1, bias=False),
#             # nn.Linear(channel, channel // reduction, bias=False),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(reduction, in_channels, kernel_size=1, bias=False),
#             nn.Softmax(dim=-2)  ################%%%%%%%%%%%%
#             # nn.Sigmoid()           ###########################
#         )
#         self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#     def forward(self, x):
#         r"""
#             Forward pass
#
#             Parameters
#             ----------
#             x: torch.Tensor, shape (B, d_in, N, K)
#
#             Returns
#             -------
#             torch.Tensor, shape (B, d_out, N, 1)
#         """
#
#         # d = torch.rand(16, 512,48)
#         # b = self.avg_pool1(d)
#
#         # computing attention scores
#         scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  #########%%%%%%%%%%%%%%%%
#
#         # a = self.avg_pool(x.permute(0,3,1,2))
#         # scores = self.score_fn(a)
#         # scores = scores.permute(0,2,3,1)
#
#         # scores = self.score_fn(self.avg_pool(x))   ############################
#
#         # sum over the neighbors
#         features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)
#         features = self.mlp(features)
#         # features = scores * x  ##traditional se
#
#         return features
#         # return self.mlp(features)  ############%%%%%%%%%%%%%


# class LocalSelfAttention2(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(LocalSelfAttention2, self).__init__()
#         self.q_conv = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#         self.k_conv = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#         self.v_conv = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#
#         self.trans_conv = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, feature):
#         q = self.q_conv(feature).permute(0, 2, 3, 1)
#         k = self.k_conv(feature).permute(0, 2, 1, 3)
#         v = self.v_conv(feature).permute(0, 2, 3, 1)
#
#         energy = torch.matmul(q, k)
#         attention = self.softmax(energy)
#         attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
#
#         context = torch.matmul(attention, v).permute(0, 3, 1, 2)
#         context = self.trans_conv(feature - context)
#         feature = feature + context
#
#         return featurecc

# class LocalSelfAttention2(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(LocalSelfAttention2, self).__init__()
#         self.q_conv = SharedMLP(in_c, 16, bn=False, activation_fn=False)
#         self.k_conv = SharedMLP(in_c, 16, bn=False, activation_fn=False)
#         self.v_conv = SharedMLP(in_c, 16, bn=False, activation_fn=False)
#
#         self.trans_conv = SharedMLP(16, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.softmax = nn.Softmax(dim=-1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, neighbour, center):
#         q = self.q_conv(neighbour).permute(0, 2, 3, 1)
#         k = self.k_conv(center).permute(0, 2, 3, 1)
#         v = self.v_conv(neighbour).permute(0, 2, 3, 1)
#
#         energy = torch.mul(q, k)
#         attention = self.sigmoid(energy)
#
#         context = torch.mul(attention, v)
#         context = self.trans_conv(context.permute(0, 3, 1, 2))
#
#         context = neighbour + context
#
#         # q = self.q_conv(feature).permute(0, 2, 3, 1)
#         # k = self.k_conv(feature).permute(0, 2, 1, 3)
#         # v = self.v_conv(feature).permute(0, 2, 3, 1)
#         #
#         # energy = torch.matmul(q, k)
#         # attention = self.softmax(energy)
#         # attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
#         #
#         # context = torch.matmul(attention, v).permute(0, 3, 1, 2)
#         # context = self.trans_conv(feature - context)
#         # feature = feature + context
#
#         return context
#
#
# class FFN(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(FFN, self).__init__()
#
#         # self.fc1 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.fc2 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # self.fc1 = SharedMLP(in_c, out_c, bn=True, activation_fn=False)
#         # self.fc2 = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#
#         self.bn1 = nn.BatchNorm2d(in_c, eps=1e-6, momentum=0.99)
#         self.fc1 = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2)
#         self.fc2 = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#         self.bn2 = nn.BatchNorm2d(out_c, eps=1e-6, momentum=0.99)
#
#     def forward(self, x):
#         x = self.bn1(x)
#         x = self.fc1(x)
#         x = self.lrelu(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#
#         # x = self.fc1(x)
#         # x = self.fc2(x)
#
#         return x


# class LocalTrans(nn.Module):
#     def __init__(self, in_c, out_c, patch_num, stage=False):
#         super(LocalTrans, self).__init__()
#
#         self.patchNum = patch_num
#         self.stage = stage
#
#         self.k1 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.q1 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.v1 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#
#         self.k2 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.q2 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.v2 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#
#         self.k3 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.q3 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.v3 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#
#         self.k4 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.q4 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.v4 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#
#         self.trans_conv = nn.Conv2d(in_c, out_c, 1, bias=False)
#
#         self.before_conv = nn.Conv2d(in_c//2, in_c, 1, bias=False)
#
#         self.ffn = FFN(out_c, out_c)
#
#
#         self.softmax = nn.Softmax(dim=-1)
#
#
#     def forward(self, features, base_features):
#
#
#         if self.stage is True:
#             features = self.before_conv(features.unsqueeze(-1)).squeeze(-1)
#             base_features = features
#
#         dist, idx = knn_point(self.patchNum, base_features.permute(0,2,1), features.permute(0,2,1))  ### B N K
#
#         center_features = features
#
#
#         #################################################################
#         neighbour = index_points(center_features.permute(0,2,1), idx[:,:,0:20:1]).permute(0,3,1,2)
#         neighbour = torch.cat((center_features.unsqueeze(-1), neighbour), dim=-1)
#
#         local_query1 = self.q1(neighbour).permute(0,2,3,1)
#         local_key1 = self.k1(neighbour).permute(0,2,1,3)
#         local_value1 = self.v1(neighbour).permute(0,2,3,1)
#
#         energy = torch.matmul(local_query1, local_key1)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention = self.softmax(energy)
#         context1 = torch.matmul(attention, local_value1).permute(0, 3, 1, 2)  ###nn.bmm
#         context1 = torch.max(context1, 3)[0]
#
#
#
#
#         #################################################################
#         neighbour = index_points(center_features.permute(0, 2, 1), idx[:, :, 0:40:2]).permute(0, 3, 1, 2)
#         neighbour = torch.cat((center_features.unsqueeze(-1), neighbour), dim=-1)
#
#         local_query2 = self.q2(neighbour).permute(0, 2, 3, 1)
#         local_key2 = self.k2(neighbour).permute(0, 2, 1, 3)
#         local_value2 = self.v2(neighbour).permute(0, 2, 3, 1)
#
#         energy = torch.matmul(local_query2, local_key2)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention = self.softmax(energy)
#         context2 = torch.matmul(attention, local_value2).permute(0, 3, 1, 2)  ###nn.bmm
#         context2 = torch.max(context2, 3)[0]
#
#
#
#         #################################################################
#         neighbour = index_points(center_features.permute(0, 2, 1), idx[:, :, 0:60:3]).permute(0, 3, 1, 2)
#         neighbour = torch.cat((center_features.unsqueeze(-1), neighbour), dim=-1)
#
#         local_query3 = self.q3(neighbour).permute(0, 2, 3, 1)
#         local_key3 = self.k3(neighbour).permute(0, 2, 1, 3)
#         local_value3 = self.v3(neighbour).permute(0, 2, 3, 1)
#
#         energy = torch.matmul(local_query3, local_key3)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention = self.softmax(energy)
#         context3 = torch.matmul(attention, local_value3).permute(0, 3, 1, 2)  ###nn.bmm
#         context3 = torch.max(context3, 3)[0]
#
#
#         ##################################################################
#         neighbour = index_points(center_features.permute(0, 2, 1), idx[:, :, 0:80:4]).permute(0, 3, 1, 2)
#         neighbour = torch.cat((center_features.unsqueeze(-1), neighbour), dim=-1)
#
#         local_query4 = self.q4(neighbour).permute(0, 2, 3, 1)
#         local_key4 = self.k4(neighbour).permute(0, 2, 1, 3)
#         local_value4 = self.v4(neighbour).permute(0, 2, 3, 1)
#
#         energy = torch.matmul(local_query4, local_key4)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention = self.softmax(energy)
#         context4 = torch.matmul(attention, local_value4).permute(0, 3, 1, 2)  ###nn.bmm
#         context4 = torch.max(context4, 3)[0]
#
#
#         context = torch.cat([context1, context2, context3, context4], dim=1).unsqueeze(-1)
#         # context = self.trans_conv(context)
#
#
#         context = center_features.unsqueeze(-1) - context
#
#
#         context = (center_features.unsqueeze(-1) + self.ffn(context)).squeeze(-1)
#
#
#
#         return context

# class NeighborAtten(nn.Module):
#     def __init__(self, in_c, out_c, patch_num):
#         super(NeighborAtten, self).__init__()
#
#         self.k = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.q = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.v = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#
#         self.patchNum = patch_num
#         self.ffn = FFN(out_c, out_c)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, features, base_features):
#         dist, idx = knn_point(self.patchNum, base_features.permute(0, 2, 1), features.permute(0, 2, 1))  ### B N K
#
#         center_features = features
#
#         local_query = self.q(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#         local_key = self.k(center_features.unsqueeze(-1)).squeeze(-1)
#         local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:20:1]).permute(0, 1, 3, 2)
#
#         local_value = self.v(center_features.unsqueeze(-1)).squeeze(-1)
#         local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:20:1])
#
#         energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention = self.softmax(energy)
#         context = torch.matmul(attention, local_value).permute(0, 3, 1, 2)  ###nn.bmm
#
#         return context
#
#
# class FuseAtten(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(FuseAtten, self).__init__()
#
#         self.mlp1 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.mlp2 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.mlp3 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.mlp4 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#     def forward(self, xyz1, point1, xyz2, point2, xyz3, point3, xyz4, point4):
#         fuse1 = torch.cat((xyz1, point1), dim=1)
#         fuse1 = fuse1.unsqueeze(-1)
#         fuse1 = self.mlp1(fuse1).squeeze(-1)
#
#         fuse2 = torch.cat((xyz2, point2), dim=1)
#         fuse2 = fuse2.unsqueeze(-1)
#         fuse2 = self.mlp2(fuse2).squeeze(-1)
#
#         fuse3 = torch.cat((xyz3, point3), dim=1)
#         fuse3 = fuse3.unsqueeze(-1)
#         fuse3 = self.mlp3(fuse3).squeeze(-1)
#
#         fuse4 = torch.cat((xyz4, point4), dim=1)
#         fuse4 = fuse4.unsqueeze(-1)
#         fuse4 = self.mlp4(fuse4).squeeze(-1)
#
#         return fuse1, fuse2, fuse3, fuse4
#
#
# class LocalTransXyz(nn.Module):
#     def __init__(self, in_c, out_c, patch_num, stage=False):
#         super(LocalTransXyz, self).__init__()
#
#         self.patchNum = patch_num
#         self.stage = stage
#
#         self.k1 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#         self.q1 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#         self.v1 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#
#         self.k2 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#         self.q2 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#         self.v2 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#
#         self.k3 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#         self.q3 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#         self.v3 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#
#         self.k4 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#         self.q4 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#         self.v4 = nn.Conv2d(in_c, out_c // 1, 1, bias=False)
#
#         # self.na1 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na2 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na3 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na4 = NeighborAtten(in_c, out_c, patch_num)
#
#         self.trans_conv = nn.Conv2d(out_c, out_c, 1, bias=False)
#
#         self.before_conv = nn.Conv2d(in_c, out_c, 1, bias=False)
#
#         self.ffn = FFN(out_c, out_c)
#         # self.gelu = nn.GELU()
#
#         self.softmax = nn.Softmax(dim=-1)
#
#         # self.norm1 = nn.LayerNorm(out_c)
#         # self.norm2 = nn.LayerNorm(out_c)
#
#     def forward(self, features, base_features, idx=None):
#
#         if self.stage is True:
#             residual = self.before_conv(features.unsqueeze(-1))
#
#         # context1 = self.na1(base_features.permute(0,2,1), features.permute(0,2,1))
#
#         if idx is not None:
#             idx = idx
#         else:
#             dist, idx = knn_point(self.patchNum, base_features.permute(0, 2, 1), features.permute(0, 2, 1))  ### B N K
#
#         center_features = features
#
#         ########################################################
#         local_query1 = self.q1(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#         local_key1 = self.k1(center_features.unsqueeze(-1)).squeeze(-1)
#         local_key1 = index_points(local_key1.permute(0, 2, 1), idx[:, :, 0:20:1]).permute(0, 1, 3, 2)
#
#         local_value1 = self.v1(center_features.unsqueeze(-1)).squeeze(-1)
#         local_value1 = index_points(local_value1.permute(0, 2, 1), idx[:, :, 0:20:1])
#
#         energy = torch.matmul(local_query1, local_key1)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention1 = self.softmax(energy)
#         context1 = torch.matmul(attention1, local_value1).permute(0, 3, 1, 2)  ###nn.bmm
#
#         #################################################################
#         local_query2 = self.q2(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#         local_key2 = self.k2(center_features.unsqueeze(-1)).squeeze(-1)
#         local_key2 = index_points(local_key2.permute(0, 2, 1), idx[:, :, 0:40:2]).permute(0, 1, 3, 2)
#
#         local_value2 = self.v2(center_features.unsqueeze(-1)).squeeze(-1)
#         local_value2 = index_points(local_value2.permute(0, 2, 1), idx[:, :, 0:40:2])
#
#         energy = torch.matmul(local_query2, local_key2)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention2 = self.softmax(energy)
#         context2 = torch.matmul(attention2, local_value2).permute(0, 3, 1, 2)  ###nn.bmm
#
#         #################################################################
#         local_query3 = self.q3(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#         local_key3 = self.k3(center_features.unsqueeze(-1)).squeeze(-1)
#         local_key3 = index_points(local_key3.permute(0, 2, 1), idx[:, :, 0:60:3]).permute(0, 1, 3, 2)
#
#         local_value3 = self.v3(center_features.unsqueeze(-1)).squeeze(-1)
#         local_value3 = index_points(local_value3.permute(0, 2, 1), idx[:, :, 0:60:3])
#
#         energy = torch.matmul(local_query3, local_key3)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention3 = self.softmax(energy)
#         context3 = torch.matmul(attention3, local_value3).permute(0, 3, 1, 2)  ###nn.bmm
#
#         ##################################################################
#         local_query4 = self.q4(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#         local_key4 = self.k4(center_features.unsqueeze(-1)).squeeze(-1)
#         local_key4 = index_points(local_key4.permute(0, 2, 1), idx[:, :, 0:80:4]).permute(0, 1, 3, 2)
#
#         local_value4 = self.v4(center_features.unsqueeze(-1)).squeeze(-1)
#         local_value4 = index_points(local_value4.permute(0, 2, 1), idx[:, :, 0:80:4])
#
#         energy = torch.matmul(local_query4, local_key4)  # + pos_encq ###nn.bmm
#         energy = energy / np.sqrt(energy.size(-1))
#
#         attention4 = self.softmax(energy)
#         context4 = torch.matmul(attention4, local_value4).permute(0, 3, 1, 2)  ###nn.bmm
#
#         # context = torch.cat([context1, context2, context3, context4], dim=1)
#         context = context1 + context2 + context3 + context4
#         context = self.trans_conv(context)
#
#         context = residual - context
#
#         context = (residual + self.ffn(context)).squeeze(-1)
#         # context = center_features + context.squeeze(-1)
#
#         return context, idx
#
#
# class LocalTrans1(nn.Module):
#     def __init__(self, in_c, out_c, patch_num, stage=False, before=False, usetanh=False):
#         super(LocalTrans1, self).__init__()
#
#         self.patchNum = patch_num
#         self.stage = stage
#         self.before = before
#         self.usetanh = usetanh
#
#         self.k1 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.q1 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.v1 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#
#         self.k2 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.q2 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.v2 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#
#         self.k3 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.q3 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.v3 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#
#         self.k4 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.q4 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.v4 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#
#         # self.na1 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na2 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na3 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na4 = NeighborAtten(in_c, out_c, patch_num)
#
#         self.trans_conv = nn.Conv2d(in_c, out_c, 1, bias=False)
#
#         self.trans_local1 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         self.trans_local2 = nn.Conv2d(in_c * 2, out_c, 1, bias=False)
#
#         # self.before_conv = nn.Conv2d(in_c//2, in_c, 1, bias=False)
#         self.before_conv = SharedMLP(in_c // 2, in_c, bn=True, activation_fn=False)
#
#         self.ffn = FFN(out_c, out_c)
#         # self.gelu = nn.GELU()
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.tanh = nn.Tanh()
#
#         # self.norm1 = nn.LayerNorm(out_c)
#         # self.norm2 = nn.LayerNorm(out_c)
#
#     def forward(self, features, base_xyz=None, xyz=None, base_features=None, FPS_index=None):
#
#         if self.before is True:
#             features = self.before_conv(features.unsqueeze(-1)).squeeze(-1)
#             # base_features = features
#
#         center_features = features
#
#         if self.stage is True:
#             base_features = self.trans_local2(base_features.unsqueeze(-1)).squeeze(-1)
#         # else:
#         #     base_features = self.trans_local2(base_features.unsqueeze(-1)).squeeze(-1)
#
#         if base_xyz is not None:
#
#             dist, idx_local = knn_point(self.patchNum, base_xyz.permute(0, 2, 1), xyz.permute(0, 2, 1))  ### B N K
#
#             ######################################
#             local_query = self.q1(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k1(base_features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx_local[:, :, 0:20:1]).permute(0, 1, 3, 2)
#
#             local_value1 = self.v1(base_features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value1.permute(0, 2, 1), idx_local[:, :, 0:20:1])
#
#             if FPS_index is not None:
#                 local_value1 = index_points(local_value1.permute(0, 2, 1), FPS_index)
#                 local_value1 = local_value1.permute(0, 2, 1)
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context1 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) + local_value1.unsqueeze(-1)  ###nn.bmm
#             # context1 = F.normalize(context1, p=2, dim=1)
#
#             ######################################
#             local_query = self.q2(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k2(base_features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx_local[:, :, 0:20:1]).permute(0, 1, 3, 2)
#
#             local_value1 = self.v2(base_features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value1.permute(0, 2, 1), idx_local[:, :, 0:20:1])
#
#             if FPS_index is not None:
#                 local_value1 = index_points(local_value1.permute(0, 2, 1), FPS_index)
#                 local_value1 = local_value1.permute(0, 2, 1)
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context2 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) + local_value1.unsqueeze(-1)  ###nn.bmm
#             # context2 = F.normalize(context2, p=2, dim=1)
#
#             ######################################
#             local_query = self.q3(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k3(base_features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx_local[:, :, 0:20:1]).permute(0, 1, 3, 2)
#
#             local_value1 = self.v3(base_features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value1.permute(0, 2, 1), idx_local[:, :, 0:20:1])
#
#             if FPS_index is not None:
#                 local_value1 = index_points(local_value1.permute(0, 2, 1), FPS_index)
#                 local_value1 = local_value1.permute(0, 2, 1)
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context3 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) + local_value1.unsqueeze(-1)  ###nn.bmm
#             # context3 = F.normalize(context3, p=2, dim=1)
#
#             ######################################
#             local_query = self.q4(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k4(base_features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx_local[:, :, 0:20:1]).permute(0, 1, 3, 2)
#
#             local_value1 = self.v4(base_features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value1.permute(0, 2, 1), idx_local[:, :, 0:20:1])
#
#             if FPS_index is not None:
#                 local_value1 = index_points(local_value1.permute(0, 2, 1), FPS_index)
#                 local_value1 = local_value1.permute(0, 2, 1)
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context4 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) + local_value1.unsqueeze(-1)  ###nn.bmm
#             # context4 = F.normalize(context4, p=2, dim=1)
#
#             context = torch.cat([context1, context2, context3, context4], dim=1)
#
#             # context = self.trans_conv(context)
#             # context = center_features.unsqueeze(-1) - context
#
#             # context = (center_features.unsqueeze(-1) + self.ffn(context)).squeeze(-1)
#
#             context = self.trans_conv(context) + center_features.unsqueeze(-1)
#             context = (context + self.ffn(context)).squeeze(-1)
#
#         else:
#
#             dist, idx = knn_point(self.patchNum, features.permute(0, 2, 1), features.permute(0, 2, 1))  ### B N K
#
#             ########################################################
#             local_query = self.q1(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k1(center_features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value1 = self.v1(center_features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value1.permute(0, 2, 1), idx[:, :, 0:16:1])
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy)  # / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context1 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) + local_value1.unsqueeze(-1)  ###nn.bmm
#             context1 = F.normalize(context1, p=2, dim=1)
#
#             # context1 = self.affine_alpha1 * Gaussian(context1) + self.affine_beta1
#             # context1 = context1.permute(0,3,1,2)
#
#             #################################################################
#             local_query = self.q2(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k2(center_features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value1 = self.v2(center_features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value1.permute(0, 2, 1), idx[:, :, 0:16:1])
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy)  # / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context2 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) + local_value1.unsqueeze(-1)  ###nn.bmm
#             context2 = F.normalize(context2, p=2, dim=1)
#
#             # context2 = self.affine_alpha2 * Gaussian(context2) + self.affine_beta2
#             # context2 = context2.permute(0,3,1,2)
#
#             #################################################################
#             local_query = self.q3(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k3(center_features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value1 = self.v3(center_features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value1.permute(0, 2, 1), idx[:, :, 0:16:1])
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy)  # / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context3 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) + local_value1.unsqueeze(-1)  ###nn.bmm
#             context3 = F.normalize(context3, p=2, dim=1)
#
#             # context3 = self.affine_alpha3 * Gaussian(context3) + self.affine_beta3
#             # context3 = context3.permute(0,3,1,2)
#
#             ##################################################################
#             local_query = self.q4(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k4(center_features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value1 = self.v4(center_features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value1.permute(0, 2, 1), idx[:, :, 0:16:1])
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy)  # / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context4 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) + local_value1.unsqueeze(-1)  ###nn.bmm
#             context4 = F.normalize(context4, p=2, dim=1)
#
#             # context4 = self.affine_alpha4 * Gaussian(context4) + self.affine_beta4
#             # context4 = context4.permute(0,3,1,2)
#
#             context = torch.cat([context1, context2, context3, context4], dim=1)
#
#             context = self.trans_conv(context)
#             # context = center_features.unsqueeze(-1) - context
#
#             context = (center_features.unsqueeze(-1) + self.ffn(context)).squeeze(-1)
#
#             # context = self.trans_conv(context) + center_features.unsqueeze(-1)
#             # context = (context + self.ffn(context)).squeeze(-1)
#
#         return context
#
#
# class offset(nn.Module):
#     def __init__(self, in_channels, out_channels, nsample):
#         super(offset, self).__init__()
#
#         self.nsample = nsample
#         self.offset_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.Tanh())
#
#         self.trans = LocalTrans(in_channels, out_channels, 16, usetanh=True, residual=True, anchorV=True)
#         self.tanh = nn.Tanh()
#
#     def forward(self, xyz, features, FPS_idx, FPS_xyz):
#         # FPS_idx = farthest_point_sample(xyz, self.nsample)
#         # FPS_xyz = index_points(xyz, FPS_idx)
#         FPS_features = index_points(features.permute(0, 2, 1), FPS_idx)
#
#         dist, group_idx = knn_point(16, xyz, FPS_xyz)
#         group_features = index_points(features.permute(0, 2, 1), group_idx).permute(0, 3, 1, 2)
#         group_features = group_features - FPS_features.permute(0, 2, 1).unsqueeze(-1)
#
#         trans = self.offset_conv(group_features).permute(0, 2, 3, 1)
#         group_xyz = index_points(xyz, group_idx)
#         group_xyz = group_xyz - FPS_xyz.unsqueeze(-2)
#         xyz_offset = (trans * group_xyz).mean(dim=-2)
#
#         # xyz_offset = self.trans(xyz=FPS_xyz.permute(0, 2, 1), base_xyz=xyz.permute(0, 2, 1), features=features, FPS_idx=FPS_idx)
#         # xyz_offset = self.tanh(xyz_offset)
#
#         FPS_xyz = FPS_xyz + xyz_offset
#
#         # group_idx = knn_point(9, xyz, FPS_xyz)
#         # group_features = index_points(features.permute(0,2,1), group_idx)
#
#         return FPS_xyz, FPS_idx


# class LocalTrans(nn.Module):
#     def __init__(self, in_c, out_c, patch_num, usetanh=False, residual=False):
#         super(LocalTrans, self).__init__()
#
#         self.patchNum = patch_num
#         self.residual = residual
#         # self.before = before
#         self.usetanh = usetanh
#         # self.first = first
#
#         self.k1 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.q1 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.v1 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#
#         self.k2 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.q2 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.v2 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#
#         self.k3 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.q3 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.v3 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#
#         self.k4 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.q4 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#         self.v4 = nn.Conv2d(in_c, out_c//4, 1, bias=False)
#
#         # self.na1 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na2 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na3 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na4 = NeighborAtten(in_c, out_c, patch_num)
#
#         # self.trans_conv_xyz = nn.Conv2d(out_c, out_c, 1, bias=False)
#         # self.trans_conv = nn.Conv2d(in_c, out_c, 1, bias=False)
#
#         # self.trans_local1 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         # self.trans_local2 = nn.Conv2d(in_c*2, out_c, 1, bias=False)
#
#
#         # self.before_conv = nn.Conv2d(in_c//2, in_c, 1, bias=False)
#         self.conv_res = SharedMLP(in_c, out_c, bn=True, activation_fn=False)
#
#         # self.ffn_xyz = FFN(out_c, out_c)
#         self.ffn = FFN(out_c, out_c)
#         # self.gelu = nn.GELU()
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.tanh = nn.Tanh()
#
#         # self.norm1 = nn.LayerNorm(out_c)
#         # self.norm2 = nn.LayerNorm(out_c)
#
#
#     def forward(self, xyz, base_xyz, features, FPS_idx=None):
#
#         # if self.before is True:
#         #     features = self.before_conv(features.unsqueeze(-1)).squeeze(-1)
#             # base_features = features
#
#         # center_features = features
#
#         # if self.stage is True:
#         #     base_features = self.trans_local2(base_features.unsqueeze(-1)).squeeze(-1)
#         # else:
#         #     base_features = self.trans_local2(base_features.unsqueeze(-1)).squeeze(-1)
#
#         # if self.first is True:
#         #
#         #     residual = xyz
#         #
#         #     if self.residual is True:
#         #         residual = self.conv_res(residual.unsqueeze(-1)).squeeze(-1)
#         #
#         #     dist, idx_local = knn_point(self.patchNum, base_xyz.permute(0, 2, 1), xyz.permute(0, 2, 1))  ### B N K
#         #
#         #     ######################################
#         #     local_query = self.q1(xyz.unsqueeze(-1)).permute(0, 2, 3, 1)
#         #
#         #     local_key = self.k1(base_xyz.unsqueeze(-1)).squeeze(-1)
#         #     local_key = index_points(local_key.permute(0, 2, 1), idx_local[:, :, 0:16:1]).permute(0, 1, 3, 2)
#         #
#         #     local_value1 = self.v1(base_xyz.unsqueeze(-1)).squeeze(-1)
#         #     local_value = index_points(local_value1.permute(0, 2, 1), idx_local[:, :, 0:16:1])
#         #
#         #     if FPS_idx is not None:
#         #         local_value1 = index_points(local_value1.permute(0,2,1), FPS_idx)
#         #         local_value1 = local_value1.permute(0,2,1)
#         #
#         #     energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#         #     energy = energy / np.sqrt(energy.size(-1))
#         #
#         #     if self.usetanh is True:
#         #         attention = self.tanh(energy) / 20
#         #     else:
#         #         attention = self.softmax(energy)
#         #
#         #     context1 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ local_value1.unsqueeze(-1)  ###nn.bmm
#         #     # context1 = F.normalize(context1, p=2, dim=1)
#         #
#         #     ######################################
#         #     local_query = self.q2(xyz.unsqueeze(-1)).permute(0, 2, 3, 1)
#         #
#         #     local_key = self.k2(base_xyz.unsqueeze(-1)).squeeze(-1)
#         #     local_key = index_points(local_key.permute(0, 2, 1), idx_local[:, :, 0:16:1]).permute(0, 1, 3, 2)
#         #
#         #     local_value1 = self.v2(base_xyz.unsqueeze(-1)).squeeze(-1)
#         #     local_value = index_points(local_value1.permute(0, 2, 1), idx_local[:, :, 0:16:1])
#         #
#         #     if FPS_idx is not None:
#         #         local_value1 = index_points(local_value1.permute(0,2,1), FPS_idx)
#         #         local_value1 = local_value1.permute(0,2,1)
#         #
#         #     energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#         #     energy = energy / np.sqrt(energy.size(-1))
#         #
#         #     if self.usetanh is True:
#         #         attention = self.tanh(energy) / 20
#         #     else:
#         #         attention = self.softmax(energy)
#         #
#         #     context2 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ local_value1.unsqueeze(-1)  ###nn.bmm
#         #     # context2 = F.normalize(context2, p=2, dim=1)
#         #
#         #     ######################################
#         #     local_query = self.q3(xyz.unsqueeze(-1)).permute(0, 2, 3, 1)
#         #
#         #     local_key = self.k3(base_xyz.unsqueeze(-1)).squeeze(-1)
#         #     local_key = index_points(local_key.permute(0, 2, 1), idx_local[:, :, 0:16:1]).permute(0, 1, 3, 2)
#         #
#         #     local_value1 = self.v3(base_xyz.unsqueeze(-1)).squeeze(-1)
#         #     local_value = index_points(local_value1.permute(0, 2, 1), idx_local[:, :, 0:16:1])
#         #
#         #     if FPS_idx is not None:
#         #         local_value1 = index_points(local_value1.permute(0,2,1), FPS_idx)
#         #         local_value1 = local_value1.permute(0,2,1)
#         #
#         #     energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#         #     energy = energy / np.sqrt(energy.size(-1))
#         #
#         #     if self.usetanh is True:
#         #         attention = self.tanh(energy) / 20
#         #     else:
#         #         attention = self.softmax(energy)
#         #
#         #     context3 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ local_value1.unsqueeze(-1)  ###nn.bmm
#         #     # context3 = F.normalize(context3, p=2, dim=1)
#         #
#         #
#         #     ######################################
#         #     local_query = self.q4(xyz.unsqueeze(-1)).permute(0, 2, 3, 1)
#         #
#         #     local_key = self.k4(base_xyz.unsqueeze(-1)).squeeze(-1)
#         #     local_key = index_points(local_key.permute(0, 2, 1), idx_local[:, :, 0:16:1]).permute(0, 1, 3, 2)
#         #
#         #     local_value1 = self.v4(base_xyz.unsqueeze(-1)).squeeze(-1)
#         #     local_value = index_points(local_value1.permute(0, 2, 1), idx_local[:, :, 0:16:1])
#         #
#         #     if FPS_idx is not None:
#         #         local_value1 = index_points(local_value1.permute(0,2,1), FPS_idx)
#         #         local_value1 = local_value1.permute(0,2,1)
#         #
#         #     energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#         #     energy = energy / np.sqrt(energy.size(-1))
#         #
#         #     if self.usetanh is True:
#         #         attention = self.tanh(energy) / 20
#         #     else:
#         #         attention = self.softmax(energy)
#         #
#         #     context4 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ local_value1.unsqueeze(-1)  ###nn.bmm
#         #     # context4 = F.normalize(context4, p=2, dim=1)
#         #
#         #
#         #     context = torch.cat([context1, context2, context3, context4], dim=1)
#         #
#         #     # context = self.trans_conv_xyz(context)
#         #     # context = center_features.unsqueeze(-1) - context
#         #
#         #     context = (residual.unsqueeze(-1) + self.ffn(context)).squeeze(-1)
#         #
#         #
#         #     # context = self.trans_conv(context) + center_features.unsqueeze(-1)
#         #     # context = (context + self.ffn(context)).squeeze(-1)
#         #
#         # else:
#         if FPS_idx is not None:
#             residual = index_points(features.permute(0, 2, 1), FPS_idx).permute(0, 2, 1)
#             center_features = residual
#
#             if self.residual is True:
#                 residual = self.conv_res(residual.unsqueeze(-1)).squeeze(-1)
#
#             dist, idx = knn_point(self.patchNum, base_xyz.permute(0, 2, 1), xyz.permute(0, 2, 1))  ### B N K
#
#             ########################################################
#             local_query = self.q1(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k1(features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value = self.v1(features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:16:1])
#             # anchor_value = self.v1(center_features.unsqueeze(-1))
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context1 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ anchor_value ###nn.bmm
#             # context1 = F.normalize(context1, p=2, dim=1)
#
#             # context1 = self.affine_alpha1 * Gaussian(context1) + self.affine_beta1
#             # context1 = context1.permute(0,3,1,2)
#
#             #################################################################
#             local_query = self.q2(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#
#             local_key = self.k2(features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value = self.v2(features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:16:1])
#             # anchor_value = self.v2(center_features.unsqueeze(-1))
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context2 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ anchor_value
#             # context2 = F.normalize(context2, p=2, dim=1)
#
#             # context2 = self.affine_alpha2 * Gaussian(context2) + self.affine_beta2
#             # context2 = context2.permute(0,3,1,2)
#
#             #################################################################
#             local_query = self.q3(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k3(features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value = self.v3(features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:16:1])
#             # anchor_value = self.v3(center_features.unsqueeze(-1))
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context3 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ anchor_value
#             # context3 = F.normalize(context3, p=2, dim=1)
#
#             # context3 = self.affine_alpha3 * Gaussian(context3) + self.affine_beta3
#             # context3 = context3.permute(0,3,1,2)
#
#             ##################################################################
#             local_query = self.q4(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k4(features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value = self.v4(features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:16:1])
#             # anchor_value = self.v4(center_features.unsqueeze(-1))
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context4 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ anchor_value
#             # context4 = F.normalize(context4, p=2, dim=1)
#
#             # context4 = self.affine_alpha4 * Gaussian(context4) + self.affine_beta4
#             # context4 = context4.permute(0,3,1,2)
#
#             context = torch.cat([context1, context2, context3, context4], dim=1)
#
#             # context = self.trans_conv(context)
#             # context = center_features.unsqueeze(-1) - context
#
#             context = (residual.unsqueeze(-1) + self.ffn(context)).squeeze(-1)
#
#             # context = self.trans_conv(context) + center_features.unsqueeze(-1)
#             # context = (context + self.ffn(context)).squeeze(-1)
#
#         else:
#             residual = features
#             center_features = residual
#
#             if self.residual is True:
#                 residual = self.conv_res(residual.unsqueeze(-1)).squeeze(-1)
#
#             dist, idx = knn_point(self.patchNum, base_xyz.permute(0, 2, 1), xyz.permute(0, 2, 1))  ### B N K
#
#             ########################################################
#             local_query = self.q1(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k1(features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value = self.v1(features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:16:1])
#             # anchor_value = self.v1(center_features.unsqueeze(-1))
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context1 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ anchor_value
#             # context1 = F.normalize(context1, p=2, dim=1)
#
#             # context1 = self.affine_alpha1 * Gaussian(context1) + self.affine_beta1
#             # context1 = context1.permute(0,3,1,2)
#
#             #################################################################
#             local_query = self.q2(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k2(features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value = self.v2(features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:16:1])
#             # anchor_value = self.v2(center_features.unsqueeze(-1))
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context2 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ anchor_value
#             # context2 = F.normalize(context2, p=2, dim=1)
#
#             # context2 = self.affine_alpha2 * Gaussian(context2) + self.affine_beta2
#             # context2 = context2.permute(0,3,1,2)
#
#             #################################################################
#             local_query = self.q3(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k3(features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value = self.v3(features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:16:1])
#             # anchor_value = self.v3(center_features.unsqueeze(-1))
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context3 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ anchor_value
#             # context3 = F.normalize(context3, p=2, dim=1)
#
#             # context3 = self.affine_alpha3 * Gaussian(context3) + self.affine_beta3
#             # context3 = context3.permute(0,3,1,2)
#
#             ##################################################################
#             local_query = self.q4(center_features.unsqueeze(-1)).permute(0, 2, 3, 1)
#
#             local_key = self.k4(features.unsqueeze(-1)).squeeze(-1)
#             local_key = index_points(local_key.permute(0, 2, 1), idx[:, :, 0:16:1]).permute(0, 1, 3, 2)
#
#             local_value = self.v4(features.unsqueeze(-1)).squeeze(-1)
#             local_value = index_points(local_value.permute(0, 2, 1), idx[:, :, 0:16:1])
#             # anchor_value = self.v4(center_features.unsqueeze(-1))
#
#             energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
#             energy = energy / np.sqrt(energy.size(-1))
#
#             if self.usetanh is True:
#                 attention = self.tanh(energy) / 20
#             else:
#                 attention = self.softmax(energy)
#
#             context4 = torch.matmul(attention, local_value).permute(0, 3, 1, 2) #+ anchor_value
#             # context4 = F.normalize(context4, p=2, dim=1)
#
#             # context4 = self.affine_alpha4 * Gaussian(context4) + self.affine_beta4
#             # context4 = context4.permute(0,3,1,2)
#
#             context = torch.cat([context1, context2, context3, context4], dim=1)
#
#             # context = self.trans_conv(context)
#             # context = center_features.unsqueeze(-1) - context
#
#             context = (residual.unsqueeze(-1) + self.ffn(context)).squeeze(-1)
#
#             # context = self.trans_conv(context) + center_features.unsqueeze(-1)
#             # context = (context + self.ffn(context)).squeeze(-1)
#
#
#         return context

# class LocalTrans(nn.Module):
#     def __init__(self, in_c, out_c, patch_num, usetanh=False, residual=False, anchorV=True, bn=True):
#         super(LocalTrans, self).__init__()
#
#         self.patchNum = patch_num
#         self.residual = residual
#         # self.before = before
#         self.usetanh = usetanh
#         self.anchorV = anchorV
#
#         # self.k1 = nn.Conv2d(in_c, out_c, 1, bias=False)
#         # self.q1 = nn.Conv2d(in_c, out_c, 1, bias=False)
#         # self.v1 = nn.Conv2d(in_c, out_c, 1, bias=False)
#
#         self.k1 = nn.Linear(in_c, in_c//4)
#         self.q1 = nn.Linear(in_c, in_c//4)
#         self.v1 = nn.Linear(in_c, out_c//4)
#
#         self.k2 = nn.Linear(in_c, in_c//4)
#         self.q2 = nn.Linear(in_c, in_c//4)
#         self.v2 = nn.Linear(in_c, out_c//4)
#
#         self.k3 = nn.Linear(in_c, in_c//4)
#         self.q3 = nn.Linear(in_c, in_c//4)
#         self.v3 = nn.Linear(in_c, out_c//4)
#
#         self.k4 = nn.Linear(in_c, in_c//4)
#         self.q4 = nn.Linear(in_c, in_c//4)
#         self.v4 = nn.Linear(in_c, out_c//4)
#
#         # self.na1 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na2 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na3 = NeighborAtten(in_c, out_c, patch_num)
#         # self.na4 = NeighborAtten(in_c, out_c, patch_num)
#
#         # self.trans_conv_xyz = nn.Conv2d(out_c, out_c, 1, bias=False)
#
#         self.trans_conv = Linear(out_c, out_c)
#
#         # self.trans_local1 = nn.Conv2d(in_c, out_c // 4, 1, bias=False)
#         # self.trans_local2 = nn.Conv2d(in_c*2, out_c, 1, bias=False)
#
#         # self.before_conv = nn.Conv2d(in_c//2, in_c, 1, bias=False)
#         self.conv_res = Linear(in_c, out_c, bn=bn)
#
#         # self.ffn_xyz = FFN(out_c, out_c)
#         # self.ffn = FFN(out_c, out_c)
#         # self.ffn = SharedMLP(out_c, out_c, bn=True, activation_fn=False
#         self.ffn = Linear(out_c, out_c, bn=bn)
#
#         self.pos_emb = Linear(3, in_c, bn=bn)
#
#         # self.gelu = nn.GELU()
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.tanh = nn.Tanh()
#
#         # self.norm1 = nn.LayerNorm(out_c)
#         # self.norm2 = nn.LayerNorm(out_c)
#
#     def forward(self, xyz, base_xyz, features, FPS_idx=None, value=None, knn=None):
#
#         pos_emb = self.pos_emb(base_xyz)
#         features = features + pos_emb
#
#         if FPS_idx is not None:
#             residual = index_points(features, FPS_idx)
#
#             # if value is not None:
#             #     residual = residual.permute(0,2,1) * value.unsqueeze(-1)  ################
#             #     residual = residual.permute(0,2,1)   ##########
#
#             center_features = residual
#
#             # if self.residual is True:
#             #     residual = self.conv_res(residual.unsqueeze(-1)).squeeze(-1)
#
#             if knn is not None:
#                 idx = knn
#             else:
#                 dist, idx = knn_point(self.patchNum, base_xyz, xyz)  ### B N K
#
#             ########################################################
#             local_query1 = self.q1(center_features).unsqueeze(-2)
#
#             local_key1 = self.k1(features)
#             local_key1 = index_points(local_key1, idx[:, :, 0:16:1]).permute(0, 1, 3, 2).contiguous()
#
#             local_value1 = self.v1(features)
#             local_value1 = index_points(local_value1, idx[:, :, 0:16:1])
#             anchor_value1 = self.v1(center_features)
#
#             energy1 = torch.matmul(local_query1, local_key1)  # + pos_encq ###nn.bmm
#             energy1 = energy1 / np.sqrt(energy1.size(-1))
#
#             if self.usetanh is True:
#                 attention1 = self.tanh(energy1) / 16
#                 # attention = F.normalize(attention, p=2, dim=3)
#             else:
#                 attention1 = self.softmax(energy1)
#
#             if self.anchorV is True:
#                 context1 = torch.matmul(attention1, local_value1).squeeze(-2) + anchor_value1
#             else:
#                 context1 = torch.matmul(attention1, local_value1).squeeze(-2)  ###nn.bmm
#
#             # context1 = self.affine_alpha1 * Gaussian(context1) + self.affine_beta1
#             # context1 = context1.permute(0,3,1,2)
#
#             #################################################################
#             local_query2 = self.q2(center_features).unsqueeze(-2)
#
#             local_key2 = self.k2(features)
#             local_key2 = index_points(local_key2, idx[:, :, 0:16:1]).permute(0, 1, 3, 2).contiguous()
#
#             local_value2 = self.v2(features)
#             local_value2 = index_points(local_value2, idx[:, :, 0:16:1])
#             anchor_value2 = self.v2(center_features)
#
#             energy2 = torch.matmul(local_query2, local_key2)  # + pos_encq ###nn.bmm
#             energy2 = energy2 / np.sqrt(energy2.size(-1))
#
#             if self.usetanh is True:
#                 attention2 = self.tanh(energy2) / 16
#                 # attention = F.normalize(attention, p=2, dim=3)
#             else:
#                 attention2 = self.softmax(energy2)
#
#             if self.anchorV is True:
#                 context2 = torch.matmul(attention2, local_value2).squeeze(-2) + anchor_value2
#             else:
#                 context2 = torch.matmul(attention2, local_value2).squeeze(-2)  ###nn.bmm
#
#             # context2 = self.affine_alpha2 * Gaussian(context2) + self.affine_beta2
#             # context2 = context2.permute(0,3,1,2)
#
#             #################################################################
#             local_query3 = self.q3(center_features).unsqueeze(-2)
#
#             local_key3 = self.k3(features)
#             local_key3 = index_points(local_key3, idx[:, :, 0:16:1]).permute(0, 1, 3, 2).contiguous()
#
#             local_value3 = self.v3(features)
#             local_value3 = index_points(local_value3, idx[:, :, 0:16:1])
#             anchor_value3 = self.v3(center_features)
#
#             energy3 = torch.matmul(local_query3, local_key3)  # + pos_encq ###nn.bmm
#             energy3 = energy3 / np.sqrt(energy3.size(-1))
#
#             if self.usetanh is True:
#                 attention3 = self.tanh(energy3) / 16
#                 # attention = F.normalize(attention, p=2, dim=3)
#             else:
#                 attention3 = self.softmax(energy3)
#
#             if self.anchorV is True:
#                 context3 = torch.matmul(attention3, local_value3).squeeze(-2) + anchor_value3
#             else:
#                 context3 = torch.matmul(attention3, local_value3).squeeze(-2)  ###nn.bmm
#
#             # context3 = self.affine_alpha3 * Gaussian(context3) + self.affine_beta3
#             # context3 = context3.permute(0,3,1,2)
#
#             ##################################################################
#             local_query4 = self.q4(center_features).unsqueeze(-2)
#
#             local_key4 = self.k4(features)
#             local_key4 = index_points(local_key4, idx[:, :, 0:16:1]).permute(0, 1, 3, 2).contiguous()
#
#             local_value4 = self.v4(features)
#             local_value4 = index_points(local_value4, idx[:, :, 0:16:1])
#             anchor_value4 = self.v4(center_features)
#
#             energy4 = torch.matmul(local_query4, local_key4)  # + pos_encq ###nn.bmm
#             energy4 = energy4 / np.sqrt(energy4.size(-1))
#
#             if self.usetanh is True:
#                 attention4 = self.tanh(energy4) / 16
#                 # attention = F.normalize(attention, p=2, dim=3)
#             else:
#                 attention4 = self.softmax(energy4)
#
#             if self.anchorV is True:
#                 context4 = torch.matmul(attention4, local_value4).squeeze(-2) + anchor_value4
#             else:
#                 context4 = torch.matmul(attention4, local_value4).squeeze(-2)  ###nn.bmm
#
#             # context4 = self.affine_alpha4 * Gaussian(context4) + self.affine_beta4
#             # context4 = context4.permute(0,3,1,2)
#
#             context = torch.cat([context1, context2, context3, context4], dim=-1)
#
#             context = self.trans_conv(context)
#
#             context = residual + self.ffn(residual - context)
#             # context = self.ffn(residual.unsqueeze(-1) - context).squeeze(-1)
#             # context = (residual.unsqueeze(-1) + context).squeeze(-1)
#
#             # context = self.trans_conv(context) + center_features.unsqueeze(-1)
#             # context = (context + self.ffn(context)).squeeze(-1)
#
#         else:
#             residual = features
#             center_features = residual
#
#             if self.residual is True:
#                 residual = self.conv_res(residual)
#
#             if knn is not None:
#                 idx = knn
#             else:
#                 dist, idx = knn_point(self.patchNum, base_xyz, xyz)  ### B N K
#
#             ########################################################
#             local_query1 = self.q1(center_features).unsqueeze(-2)
#
#             local_key1 = self.k1(features)
#             local_key1 = index_points(local_key1, idx[:, :, 0:16:1]).permute(0, 1, 3, 2).contiguous()
#
#             local_value1 = self.v1(features)
#             local_value1 = index_points(local_value1, idx[:, :, 0:16:1])
#             anchor_value1 = self.v1(center_features)
#
#             energy1 = torch.matmul(local_query1, local_key1)  # + pos_encq ###nn.bmm
#             energy1 = energy1 / np.sqrt(energy1.size(-1))
#
#             if self.usetanh is True:
#                 attention1 = self.tanh(energy1) / 16
#                 # attention = F.normalize(attention, p=2, dim=3)
#             else:
#                 attention1 = self.softmax(energy1)
#
#             if self.anchorV is True:
#                 context1 = torch.matmul(attention1, local_value1).squeeze(-2) + anchor_value1
#             else:
#                 context1 = torch.matmul(attention1, local_value1).squeeze(-2)  ###nn.bmm
#
#             # context1 = self.affine_alpha1 * Gaussian(context1) + self.affine_beta1
#             # context1 = context1.permute(0,3,1,2)
#
#             #################################################################
#             local_query2 = self.q2(center_features).unsqueeze(-2)
#
#             local_key2 = self.k2(features)
#             local_key2 = index_points(local_key2, idx[:, :, 0:16:1]).permute(0, 1, 3, 2).contiguous()
#
#             local_value2 = self.v2(features)
#             local_value2 = index_points(local_value2, idx[:, :, 0:16:1])
#             anchor_value2 = self.v2(center_features)
#
#             energy2 = torch.matmul(local_query2, local_key2)  # + pos_encq ###nn.bmm
#             energy2 = energy2 / np.sqrt(energy2.size(-1))
#
#             if self.usetanh is True:
#                 attention2 = self.tanh(energy2) / 16
#                 # attention = F.normalize(attention, p=2, dim=3)
#             else:
#                 attention2 = self.softmax(energy2)
#
#             if self.anchorV is True:
#                 context2 = torch.matmul(attention2, local_value2).squeeze(-2) + anchor_value2
#             else:
#                 context2 = torch.matmul(attention2, local_value2).squeeze(-2)  ###nn.bmm
#
#             # context2 = self.affine_alpha2 * Gaussian(context2) + self.affine_beta2
#             # context2 = context2.permute(0,3,1,2)
#
#             #################################################################
#             local_query3 = self.q3(center_features).unsqueeze(-2)
#
#             local_key3 = self.k3(features)
#             local_key3 = index_points(local_key3, idx[:, :, 0:16:1]).permute(0, 1, 3, 2).contiguous()
#
#             local_value3 = self.v3(features)
#             local_value3 = index_points(local_value3, idx[:, :, 0:16:1])
#             anchor_value3 = self.v3(center_features)
#
#             energy3 = torch.matmul(local_query3, local_key3)  # + pos_encq ###nn.bmm
#             energy3 = energy3 / np.sqrt(energy3.size(-1))
#
#             if self.usetanh is True:
#                 attention3 = self.tanh(energy3) / 16
#                 # attention = F.normalize(attention, p=2, dim=3)
#             else:
#                 attention3 = self.softmax(energy3)
#
#             if self.anchorV is True:
#                 context3 = torch.matmul(attention3, local_value3).squeeze(-2) + anchor_value3
#             else:
#                 context3 = torch.matmul(attention3, local_value3).squeeze(-2)  ###nn.bmm
#
#             # context3 = self.affine_alpha3 * Gaussian(context3) + self.affine_beta3
#             # context3 = context3.permute(0,3,1,2)
#
#             ##################################################################
#             local_query4 = self.q4(center_features).unsqueeze(-2)
#
#             local_key4 = self.k4(features)
#             local_key4 = index_points(local_key4, idx[:, :, 0:16:1]).permute(0, 1, 3, 2).contiguous()
#
#             local_value4 = self.v4(features)
#             local_value4 = index_points(local_value4, idx[:, :, 0:16:1])
#             anchor_value4 = self.v4(center_features)
#
#             energy4 = torch.matmul(local_query4, local_key4)  # + pos_encq ###nn.bmm
#             energy4 = energy4 / np.sqrt(energy4.size(-1))
#
#             if self.usetanh is True:
#                 attention4 = self.tanh(energy4) / 16
#                 # attention = F.normalize(attention, p=2, dim=3)
#             else:
#                 attention4 = self.softmax(energy4)
#
#             if self.anchorV is True:
#                 context4 = torch.matmul(attention4, local_value4).squeeze(-2) + anchor_value4
#             else:
#                 context4 = torch.matmul(attention4, local_value4).squeeze(-2)  ###nn.bmm
#
#             # context4 = self.affine_alpha4 * Gaussian(context4) + self.affine_beta4
#             # context4 = context4.permute(0,3,1,2)
#
#             context = torch.cat([context1, context2, context3, context4], dim=-1)
#
#             context = self.trans_conv(context)
#
#             context = residual + self.ffn(residual - context)
#             # context = self.ffn(residual.unsqueeze(-1) - context).squeeze(-1)
#             # context = (residual.unsqueeze(-1) + context).squeeze(-1)
#
#             # context = self.trans_conv(context) + center_features.unsqueeze(-1)
#             # context = (context + self.ffn(context)).squeeze(-1)
#
#         return context
#
#
# class GetGraph(nn.Module):
#     def __init__(self, in_c, out_c, sample_num, residual=False):
#         super(GetGraph, self).__init__()
#
#         self.mlp1 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.mlp2 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.sample_num = sample_num
#
#         self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_c, eps=1e-6, momentum=0.99)
#
#         self.conv2 = nn.Conv2d(out_c, out_c, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_c, eps=1e-6, momentum=0.99)
#
#         self.conv3 = nn.Conv2d(in_c, out_c, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_c, eps=1e-6, momentum=0.99)
#
#         self.residual = residual
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2)
#
#     def forward(self, features):
#         # features_center = self.mlp1(features.unsqueeze(-1))
#         residual = features.unsqueeze(-1)
#
#         if self.residual is True:
#             residual = self.bn3(self.conv3(residual))
#
#         features_center = features.unsqueeze(-1)
#         features = features.permute(0, 2, 1).contiguous()  ## B N C
#
#         dist, idx = knn_point(self.sample_num, features, features)  ### B N K
#
#         neighbors_features = index_points(features, idx)
#         neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
#
#         features = torch.cat((features_center, neighbors_features), dim=-1)
#         features = features.sum(dim=-1, keepdim=True)
#
#         features = self.conv1(features)
#         # features = features.sum(dim=-1, keepdim=True)
#         features = self.bn1(features)
#         features = self.lrelu(features)
#
#         features = self.conv2(features)
#         features = self.bn2(features)
#
#         features = features + residual
#
#         features = self.lrelu(features).squeeze(-1)
#
#         # features = torch.cat((neighbors_features - features_center, features_center), dim=1)
#         # features = self.mlp(features)
#         # features = torch.max(features, 3)[0]
#
#         # neighbors_features = self.mlp2(neighbors_features)
#
#         # neighbors_features = torch.max(neighbors_features, 3)[0].unsqueeze(-1)
#
#         # features = (neighbors_features - features_center).squeeze(-1).contiguous()
#
#         return features
#
#
# class LocalAggregation(nn.Module):
#     def __init__(self, in_c, out_c, sample_num_list, affine_c=64):
#         super(LocalAggregation, self).__init__()
#
#         self.sample_num_list = sample_num_list
#         # self.feature_mlp = SharedMLP(in_c *2, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # self.final_feature_mlp = SharedMLP(in_c*3, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # self.coord_mlp = SharedMLP(10, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.coord_pool = AttentivePooling(in_c, in_c, in_c//4)
#
#         # self.coord_mlp2 = SharedMLP(10, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.coord_pool2 = AttentivePooling(in_c, in_c, in_c//4)
#
#         # self.feature_pool = AttentivePooling(in_c, in_c, in_c//4)
#         # self.feature_pool2 = AttentivePooling(d1, d1, d1 // 4)
#
#         # self.mlp = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.mlp = conv1x1(in_c, out_c)
#         self.mlp = Linear(in_c, out_c, bn=False)
#
#         # self.mlp1 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.mlp2 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.mlp3 = SharedMLP(13, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.mlp4 = SharedMLP(in_c + 12, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.mlp4 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.short_cut = SharedMLP(in_c, in_c, bn=True)
#
#         # self.local1 = BasicBlock(in_c*2+1, in_c)
#         # self.local2 = BasicBlock(in_c*2+1, in_c)
#
#         # self.coord_pool = AttentivePooling(out_c, out_c, 4)
#         # self.feature_pool = AttentivePooling(in_c, out_c, 4)
#         # self.localTrans1 = LocalTrans(out_c, out_c)
#         # self.localTrans2 = LocalTrans(in_c, out_c)
#         # self.spatial_attention1 = SpatialAttention(out_c, out_c)
#         # self.spatial_attention2 = SpatialAttention(out_c, out_c)
#         # self.conv_blocks = nn.ModuleList()
#         # self.bn_blocks = nn.ModuleList()
#         # for i in range(len(sample_num_list)):
#         #     convs = nn.ModuleList()
#         #     bns = nn.ModuleList()
#         #     # last_channel = in_channel + 3
#         #     # for out_channel in sample_num_list[i]:
#         #     convs.append(nn.Conv2d(in_c * 2, in_c, 1))
#         #     bns.append(nn.BatchNorm2d(in_c))
#         #     # last_channel = out_channel
#         #     self.conv_blocks.append(convs)
#         #     self.bn_blocks.append(bns)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2)
#
#         # a = torch.randn([1,1,1,20], dtype=torch.float32)
#         # b = torch.randn([1,1,1,20], dtype=torch.float32)
#         #
#         # self.neighbourPara = nn.Parameter(a)
#         # self.neighbourPara2 = nn.Parameter(b)
#
#         # self.affine_alpha = nn.Parameter(torch.ones([1,1,1,affine_c]))
#         # self.affine_beta = nn.Parameter(torch.zeros([1,1,1,affine_c]))
#         #
#         # self.affine_alpha2 = nn.Parameter(torch.ones([1,1,1,affine_c]))
#         # self.affine_beta2 = nn.Parameter(torch.zeros([1,1,1,affine_c]))
#         #
#         # self.affine_alpha3 = nn.Parameter(torch.ones([1,1,1,3]))
#         # self.affine_beta3 = nn.Parameter(torch.zeros([1,1,1,3]))
#
#     def forward(self, features, xyz=None, norm=None):
#
#         features = features.permute(0, 2, 1).contiguous()  ## B N C
#         if norm is not None:
#             norm = norm.permute(0, 2, 1).contiguous()  ## B N C
#         # upper_coords = upper_coords.permute(0, 2, 1)  ## B N C
#         # coords = coords.permute(0, 2, 1)  ## B N C
#
#         S = features.size(2)
#
#         for i, sample_num in enumerate(self.sample_num_list):
#
#             dist, idx = knn_point(sample_num, features, features)  ### B N K
#
#             # a = knn(features, sample_num)
#             B, N, K = idx.size()
#
#             if S == 3:
#                 features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()
#
#                 neighbors_features = index_points(features, idx)
#                 neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
#
#                 # neighbors_features = self.localTrans1(neighbors_features)    ###############
#
#                 dist = torch.sqrt(torch.sum((neighbors_features.permute(0, 2, 3,
#                                                                         1).contiguous() - features_centre.permute(0, 2,
#                                                                                                                   3,
#                                                                                                                   1).contiguous()) ** 2,
#                                             dim=-1))
#                 # dist_norm = torch.sqrt(torch.sum((neighbors_norm.permute(0, 2, 3, 1).contiguous() - norm_center.permute(0, 2, 3, 1).contiguous()) ** 2, dim=-1))
#                 # dist = torch.sum(torch.sqrt((neighbors_features.permute(0, 2, 3, 1).contiguous() - features_centre.permute(0, 2, 3, 1).contiguous())**2), dim=-1)
#
#                 x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_features, features_centre)
#
#                 # if norm is not None:
#                 #     neighbours_norm = index_points(norm, idx).permute(0, 3, 1, 2).contiguous()
#                 #     features = torch.cat((neighbors_features - features_centre, neighbors_features, x_alpha, x_beta, y_alpha,
#                 #          y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous(), neighbours_norm), dim=1)
#                 # else:
#
#                 # affine_features = self.affine_alpha * Gaussian(neighbors_features - features_centre) + self.affine_beta
#                 # affine_features = affine_features.permute(0,3,1,2)
#                 # neighbors_features = affine_features
#
#                 features = torch.cat(
#                     (neighbors_features - features_centre, neighbors_features, x_alpha, x_beta, y_alpha,
#                      y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous()), dim=1)
#
#                 features = features.permute(0, 2, 3, 1).contiguous()
#                 B, N, K, C = features.shape
#                 features = features.reshape(B, -1, C).contiguous()
#                 features = self.mlp(features)
#                 features = features.reshape(B, N, K, -1).contiguous()
#
#                 # features = torch.mul(self.neighbourPara, features) #######
#                 # features = torch.sum(features, 3)   ###########
#
#                 features = torch.max(features, 2)[0]
#                 # features = features.permute(0,2,1).contiguous()
#
#                 # features = self.localTrans1(features)
#
#             elif xyz is not None:
#                 ###################  xyz
#                 xyz_center = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K).contiguous()
#
#                 neighbors_xyz = index_points(xyz, idx)
#                 neighbors_xyz = neighbors_xyz.permute(0, 3, 1, 2).contiguous()
#
#                 x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_xyz, xyz_center)
#
#                 # affine_xyz = self.affine_alpha3 * Gaussian(neighbors_xyz-xyz_center) + self.affine_beta3   #################
#                 # affine_xyz = affine_xyz.permute(0,3,1,2)                          ############################
#                 # neighbors_xyz = affine_features                                ########################
#
#                 features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center, x_alpha, x_beta, y_alpha,
#                                           y_beta, z_alpha, z_beta), dim=1)
#
#                 features_centre = self.mlp1(features.transpose(-2, -1).unsqueeze(-1).contiguous())
#                 neighbors_features = index_points(features, idx)
#                 neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
#
#                 # neighbors_features = self.localTrans2(neighbors_features)  ##########
#
#                 # affine_features = self.affine_alpha2 * Gaussian(neighbors_features-features_centre) + self.affine_beta2   #################
#                 # affine_features = affine_features.permute(0,3,1,2)                          ############################
#                 # neighbors_features = affine_features                                ########################
#
#                 neighbors_features = self.mlp4(torch.cat((neighbors_features, features_xyz), dim=1))
#                 # neighbors_features = self.mlp4(neighbors_features)
#
#                 neighbors_features = torch.max(neighbors_features, 3)[0].unsqueeze(-1)
#
#                 # neighbors_features = torch.mul(self.neighbourPara2, neighbors_features) #######
#                 # neighbors_features = torch.sum(neighbors_features, 3).unsqueeze(-1)   ###########
#
#                 features = (neighbors_features - features_centre).squeeze(-1).contiguous()
#
#                 # features = self.localTrans2(features)
#
#         return features
#
#
# class LocalAggregationPart(nn.Module):
#     def __init__(self, in_c, out_c, sample_num_list):
#         super(LocalAggregationPart, self).__init__()
#
#         self.sample_num_list = sample_num_list
#         # self.feature_mlp = SharedMLP(in_c *2, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # self.final_feature_mlp = SharedMLP(in_c*3, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # self.coord_mlp = SharedMLP(10, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.coord_pool = AttentivePooling(in_c, in_c, in_c//4)
#
#         # self.coord_mlp2 = SharedMLP(10, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.coord_pool2 = AttentivePooling(in_c, in_c, in_c//4)
#
#         # self.feature_pool = AttentivePooling(in_c, in_c, in_c//4)
#         # self.feature_pool2 = AttentivePooling(d1, d1, d1 // 4)
#         self.mlp = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.mlp1 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.mlp2 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.mlp3 = SharedMLP(13, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.mlp4 = SharedMLP(in_c + 12, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.short_cut = SharedMLP(in_c, in_c, bn=True)
#
#         # self.local1 = BasicBlock(in_c*2+1, in_c)
#         # self.local2 = BasicBlock(in_c*2+1, in_c)
#
#         self.coord_pool = AttentivePooling(out_c, out_c, 4)
#         self.feature_pool = AttentivePooling(in_c, out_c, 4)
#
#         # self.conv_blocks = nn.ModuleList()
#         # self.bn_blocks = nn.ModuleList()
#         # for i in range(len(sample_num_list)):
#         #     convs = nn.ModuleList()
#         #     bns = nn.ModuleList()
#         #     # last_channel = in_channel + 3
#         #     # for out_channel in sample_num_list[i]:
#         #     convs.append(nn.Conv2d(in_c * 2, in_c, 1))
#         #     bns.append(nn.BatchNorm2d(in_c))
#         #     # last_channel = out_channel
#         #     self.conv_blocks.append(convs)
#         #     self.bn_blocks.append(bns)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2)
#
#     def forward(self, features, xyz=None, norm=None):
#
#         features = features.permute(0, 2, 1).contiguous()  ## B N C
#         if norm is not None:
#             norm = norm.permute(0, 2, 1).contiguous()  ## B N C
#         # upper_coords = upper_coords.permute(0, 2, 1)  ## B N C
#         # coords = coords.permute(0, 2, 1)  ## B N C
#
#         S = features.size(2)
#
#         for i, sample_num in enumerate(self.sample_num_list):
#
#             dist, idx = knn_point(sample_num, features, features)  ### B N K
#
#             # a = knn(features, sample_num)
#             B, N, K = idx.size()
#
#             if S == 3:
#                 features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()
#
#                 neighbors_features = index_points(features, idx)
#                 neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
#
#                 dist = torch.sqrt(torch.sum((neighbors_features.permute(0, 2, 3,
#                                                                         1).contiguous() - features_centre.permute(0, 2,
#                                                                                                                   3,
#                                                                                                                   1).contiguous()) ** 2,
#                                             dim=-1))
#                 # dist = torch.sum(torch.sqrt((neighbors_features.permute(0, 2, 3, 1).contiguous() - features_centre.permute(0, 2, 3, 1).contiguous())**2), dim=-1)
#
#                 x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_features, features_centre)
#
#                 # if norm is not None:
#                 #     neighbours_norm = index_points(norm, idx).permute(0, 3, 1, 2).contiguous()
#                 #     features = torch.cat((neighbors_features - features_centre, neighbors_features, x_alpha, x_beta, y_alpha,
#                 #          y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous(), neighbours_norm), dim=1)
#                 # else:
#                 features = torch.cat(
#                     (neighbors_features - features_centre, neighbors_features, x_alpha, x_beta, y_alpha,
#                      y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous()), dim=1)
#                 # features = torch.cat((neighbors_features - features_centre, neighbors_features, dist.unsqueeze(-3).contiguous()), dim=1)
#                 # features_concat = torch.cat((neighbors_features - features_centre, features_centre, dist.unsqueeze(-3)), dim=1)
#                 # features_concat_list.append(features_concat)
#
#                 features = self.mlp(features)
#                 features = torch.max(features, 3)[0]
#                 # features = self.coord_pool(features).squeeze(-1)
#             elif xyz is not None:
#                 ###################  xyz
#                 xyz_center = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K).contiguous()
#
#                 neighbors_xyz = index_points(xyz, idx)
#                 neighbors_xyz = neighbors_xyz.permute(0, 3, 1, 2).contiguous()
#
#                 # dist_xyz = torch.sum(torch.sqrt((neighbors_xyz.permute(0, 2, 3, 1).contiguous() - xyz_center.permute(0, 2, 3, 1).contiguous())**2), dim=-1)
#                 x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_xyz, xyz_center)
#
#                 # if norm is not None:
#                 #     neighbours_norm = index_points(norm, idx).permute(0, 3, 1, 2).contiguous()
#                 #     features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center, x_alpha, x_beta, y_alpha,
#                 #                               y_beta, z_alpha, z_beta, neighbours_norm), dim=1)
#                 # else:
#                 features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center, x_alpha, x_beta, y_alpha,
#                                           y_beta, z_alpha, z_beta), dim=1)
#                 # features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center), dim=1)
#
#                 #################### feature
#                 # features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()
#                 #
#                 # neighbors_features = index_points(features, idx)
#                 # neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
#                 #
#                 # features = torch.cat((neighbors_features - features_centre, features_centre, dist.unsqueeze(-3).contiguous()), dim=1)
#                 #
#                 # features = self.mlp4(features)
#                 # features = torch.max(features, 3)[0]
#
#                 features_centre = self.mlp1(features.transpose(-2, -1).unsqueeze(-1).contiguous())
#                 neighbors_features = index_points(features, idx)
#                 neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
#                 neighbors_features = self.mlp4(torch.cat((neighbors_features, features_xyz), dim=1))
#                 neighbors_features = torch.max(neighbors_features, 3)[0].unsqueeze(-1)
#                 # neighbors_features = self.feature_pool(neighbors_features)
#
#                 features = (neighbors_features - features_centre).squeeze(-1).contiguous()
#
#         return features
#
#
# class Discriminator(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.f_k = nn.Bilinear(dim, dim, 1)
#         for m in self.modules():
#             self.weights_init(m)
#
#         # self.mlp = nn.Conv2d(dim*2, dim, 1, bias=False)    #############################
#         # self.lrelu = nn.LeakyReLU(negative_slope=0.2)    #######################
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Bilinear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
#
#         B, N, C = c.shape  ##########################
#
#         # sc_1 = self.mlp(torch.cat((h_pl, c), dim=1))    ##############################
#         # sc_2 = self.mlp(torch.cat((h_mi, c), dim=1))    ##############################
#         #
#         # sc_1 = sc_1.squeeze(-1).permute(0,2,1).reshape(-1, C)    ##########################
#         # sc_2 = sc_2.squeeze(-1).permute(0,2,1).reshape(-1, C)    ##########################
#         #
#         # output = torch.squeeze(self.f_k(sc_1, sc_2), -2)    ############################
#         # output = output.reshape(B, N, -1)               ############################
#         #
#         # return output       ############################
#
#         c_x = c
#
#         c_x = c_x.reshape(-1, C)  ##########################
#         h_pl = h_pl.reshape(-1, C)  ##########################
#         h_mi = h_mi.reshape(-1, C)  ##########################
#
#         sc_1 = torch.squeeze(self.f_k(h_pl, c_x), -2)
#         sc_2 = torch.squeeze(self.f_k(h_mi, c_x), -2)
#
#         sc_1 = sc_1.reshape(B, N, -1)  ############################
#         sc_2 = sc_2.reshape(B, N, -1)  ############################
#
#         if s_bias1 is not None:
#             sc_1 += s_bias1
#         if s_bias2 is not None:
#             sc_2 += s_bias2
#
#         # logits = torch.cat((sc_1, sc_2), 0).squeeze(-1)
#         # v = logits.shape[0]
#
#         logits = torch.cat((sc_1, sc_2), 1).squeeze(-1)  ###############
#         v = logits.shape[1]  ###################
#
#         # return logits, logits[:v // 2]
#         return logits, logits[:, :v // 2]  ###############
#
#
# class GCN(nn.Module):
#
#     def __init__(self, in_dim, out_dim):
#         super(GCN, self).__init__()
#         # self.proj = nn.Linear(in_dim, out_dim)
#         self.proj = conv1x1(in_dim, out_dim)  #############################
#         self.drop = nn.Dropout(p=0.5)
#
#     def forward(self, A, X, act=None):
#         X = X.squeeze(-1).permute(0, 2, 1)  ############################
#         X = self.drop(X)
#         X = torch.matmul(A, X)
#         X = X.permute(0, 2, 1).unsqueeze(-1)  ##########################
#         X = self.proj(X)
#         if act is not None:
#             X = act(X)
#         return X
#
#
# class MLP(nn.Module):
#     def __init__(self, in_ft, out_ft, act='prelu', bias=True):
#         super().__init__()
#         self.fc = nn.Linear(in_ft, out_ft, bias=bias)
#         self.act = nn.PReLU() if act == 'prelu' else act
#
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_ft))
#             self.bias.data.fill_(0.0)
#         else:
#             self.register_parameter('bias', None)
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, x):
#         x_fts = self.fc(x)
#         if self.bias is not None:
#             x_fts += self.bias
#         return self.act(x_fts)
#
#
# class IndexSelect(nn.Module):
#
#     # def __init__(self, k, n_h, act,  R=1):
#     def __init__(self, dim, knn):  ####################
#         super().__init__()
#         # self.k = k
#         # self.k = nsample   ##################
#         self.k_num = knn  ########################
#         # self.adj_shape = adj_shape    #########################
#         # self.R = R
#         self.sigm = nn.Sigmoid()
#         # self.fc = MLP(dim, dim, act)
#         self.fc = Linear(dim, dim, bn=False)
#
#         # self.lrelu = nn.LeakyReLU(negative_slope=0.2)    #######################
#
#         self.disc = Discriminator(dim)
#         # self.gcn1 = GCN(dim, dim)
#         self.Trans = LocalTrans(dim, dim, knn, usetanh=False, anchorV=False, bn=False)
#
#         # self.conv1x1 = conv1x1(dim, dim*2)
#         # self.conv1x1 = nn.Conv2d(dim, dim*2, 1, bias=False)     ########################
#
#     def forward(self, xyz, features, samp_bias1=None, samp_bias2=None):
#         _, N, _ = xyz.shape
#         _, idx = knn_point(self.k_num, xyz, xyz)  #####################################
#         A = F.one_hot(idx, num_classes=N).sum(dim=-2).float()  ################################
#         # A = output.permute(0,2,1).float()                        #########################
#         # A = (A + A.permute(0,2,1))/2
#
#         seq1 = features
#         # seq1 = features.permute(0,2,1)
#         seq2 = torch.zeros_like(seq1)
#         seq2 = seq1[:, torch.randperm(seq1.shape[1]), :]
#
#         h_1 = self.fc(seq1)
#         h_2 = self.fc(seq2)
#
#         # h_n1 = self.gcn1(A, h_1)
#         h_n1 = self.Trans(xyz=xyz, base_xyz=xyz, features=h_1, knn=idx)  #############
#
#         # X = self.sigm(h_n1)
#         X = h_n1
#         ret, ret_true = self.disc(X, h_1, h_2, samp_bias1, samp_bias2)
#         scores = self.sigm(ret_true).squeeze()
#         # num_nodes = A.shape[0]
#         num_nodes = A.shape[1] // 2  ############################
#         values, idx = torch.topk(scores, int(num_nodes))
#
#         # feature_pool = self.fc1(features.unsqueeze(-1))                    ############################
#
#         feature_pool = torch.mul(features, scores.unsqueeze(-1))  ################
#         # feature_pool = h_n1.squeeze(-1).permute(0,2,1) * scores.unsqueeze(-1)    ############################
#         feature_pool = index_points(feature_pool, idx)  ############################
#         # feature_pool = self.conv1x1(feature_pool.unsqueeze(-1)).squeeze(-1)       ############################
#
#         xyz_pool = index_points(xyz, idx)  ############################
#
#         # values1, idx1 = values[:int(self.k * num_nodes)], idx[:int(self.k * num_nodes)]
#
#         # values0, idx0 = values[int(self.k * num_nodes):], idx[int(self.k * num_nodes):]
#
#         # return ret, values1, idx1, idx0, h_n1
#         # return ret, values, idx   #########################
#         return ret, xyz_pool, feature_pool, idx  #########################
#
# class Upsample(nn.Module):
#     def __init__(self, in_c, out_c, knn):
#         super(Upsample, self).__init__()
#
#         self.fc = Linear(in_c, out_c, bn=False, act=False)
#         self.k_num = knn
#         # self.lrelu = nn.LeakyReLU(negative_slope=0.2)    #######################
#
#     def forward(self, base_xyz, base_features, features, down_idx):
#
#         B, N, C = base_features.shape
#         _, _, S = features.shape
#         ratio = S//C
#
#         x = torch.zeros_like(base_features).repeat(1,1,ratio)
#
#         dists, idx = knn_point(self.k_num, base_xyz, base_xyz)  #####################################
#         x = index_points2(x, down_idx, features)
#
#         # x = index_points(x, idx)
#
#         # dists = square_distance(xyz1, xyz2)
#         # dists, idx = dists.sort(dim=-1)
#         # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
#
#
#         # dist_recip = index_points3(dist_recip, down_idx)
#
#         # dist_recip1 = index_points(dist_recip1, down_idx)
#         # dist_recip = torch.zeros_like(dists)
#         # dist_recip = index_points2(dist_recip, down_idx, dist_recip1)
#
#
#         for i in range(3):
#             dist_recip = 1.0 / (dists + 1e-8)
#             mask = index_points(x, idx)
#             mask = torch.sum(mask, dim=-1)
#
#             mask = (mask > 1e-6).float() + (mask < -1e-6).float()
#             dist_recip = dist_recip * mask
#
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / (norm + 1e-8)
#
#             x = torch.sum(index_points(x, idx) * weight.view(B, N, self.k_num, 1), dim=2)
#             x = index_points2(x, down_idx, features)
#
#         x = self.fc(x)
#
#         return x
#
#
#
# class Fuse_Stage(nn.Module):
#     def __init__(self, b1_C, b2_C, b3_C, b4_C):
#         super(Fuse_Stage, self).__init__()
#
#         self.stage_12 = Linear(b1_C, b2_C, bn=False, act=False)
#         self.stage_13 = Linear(b1_C, b3_C, bn=False, act=False)
#         self.stage_23 = Linear(b2_C, b3_C, bn=False, act=False)
#         self.stage_14 = Linear(b1_C, b4_C, bn=False, act=False)
#         self.stage_24 = Linear(b2_C, b4_C, bn=False, act=False)
#         self.stage_34 = Linear(b3_C, b4_C, bn=False, act=False)
#
#         # self.stage_12 = SharedMLP(b1_C, b2_C, bn=True, activation_fn=False)
#         # self.stage_13 = SharedMLP(b1_C, b3_C, bn=True, activation_fn=False)
#         # self.stage_23 = SharedMLP(b2_C, b3_C, bn=True, activation_fn=False)
#         # self.stage_14 = SharedMLP(b1_C, b4_C, bn=True, activation_fn=False)
#         # self.stage_24 = SharedMLP(b2_C, b4_C, bn=True, activation_fn=False)
#         # self.stage_34 = SharedMLP(b3_C, b4_C, bn=True, activation_fn=False)
#
#         # self.stage_fuse21 = PointNetFeaturePropagation(b2_C, [b1_C, ])
#         # self.stage_fuse31 = PointNetFeaturePropagation(b3_C, [b1_C, ])
#         # self.stage_fuse32 = PointNetFeaturePropagation(b3_C, [b2_C, ])
#         # self.stage_fuse41 = PointNetFeaturePropagation(b4_C, [b1_C, ])
#         # self.stage_fuse42 = PointNetFeaturePropagation(b4_C, [b2_C, ])
#         # self.stage_fuse43 = PointNetFeaturePropagation(b4_C, [b3_C, ])
#
#         # self.stage_fuse21 = PointNetFeaturePropagation(b2_C + b1_C, [b1_C, ])
#         # self.stage_fuse31 = PointNetFeaturePropagation(b3_C + b1_C, [b1_C, ])
#         # self.stage_fuse32 = PointNetFeaturePropagation(b3_C + b2_C, [b2_C, ])
#         # self.stage_fuse41 = PointNetFeaturePropagation(b4_C + b1_C, [b1_C, ])
#         # self.stage_fuse42 = PointNetFeaturePropagation(b4_C + b2_C, [b2_C, ])
#         # self.stage_fuse43 = PointNetFeaturePropagation(b4_C + b3_C, [b3_C, ])
#
#         self.stage_fuse21 = Upsample(b2_C, b1_C, 8)
#         self.stage_fuse31 = Upsample(b3_C, b1_C, 16)
#         self.stage_fuse32 = Upsample(b3_C, b2_C, 8)
#         self.stage_fuse41 = Upsample(b4_C, b1_C, 32)
#         self.stage_fuse42 = Upsample(b4_C, b2_C, 16)
#         self.stage_fuse43 = Upsample(b4_C, b3_C, 8)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2)
#
#     def forward(self, b1_f, b2_f, b2_idx, b1_xyz, b2_xyz, b3_f=None, b4_f=None, b3_idx=None, b4_idx=None, b3_xyz=None, b4_xyz=None):
#         xyz = 0
#
#         if b3_f is None and b4_f is None:
#             # temp21_points = self.stage_fuse21(b1_xyz, b2_xyz, b1_f, b2_f)
#             temp21_points = self.stage_fuse21(b1_xyz, b1_f, b2_f, b2_idx)   ######
#             branch1_points = self.lrelu(b1_f + temp21_points)
#
#             temp12_points = self.stage_12(index_points(b1_f, b2_idx))
#             branch2_points_FP = self.lrelu(b2_f + temp12_points)
#
#             branch3_points_FP = 0
#             branch4_points_FP = 0
#
#         elif b3_f is not None and b4_f is None:
#             # temp31_points = self.stage_fuse31(b1_xyz, b3_xyz, b1_f, b3_f)
#             # temp21_points = self.stage_fuse21(b1_xyz, b2_xyz, b1_f, b2_f)
#             temp31_points = self.stage_fuse31(b1_xyz, b1_f, b3_f, b3_idx)   #######
#             temp21_points = self.stage_fuse21(b1_xyz, b1_f, b2_f, b2_idx)   #######
#             branch1_points = self.lrelu(b1_f + temp21_points + temp31_points)
#
#             # temp32_points = self.stage_fuse32(b2_xyz, b3_xyz, b2_f, b3_f)
#             temp32_points = self.stage_fuse32(b2_xyz, b2_f, b3_f, b3_idx)   #######
#             temp12_points = self.stage_12(index_points(b1_f, b2_idx))
#             branch2_points_FP = self.lrelu(b2_f + temp12_points + temp32_points)
#
#             temp13_points = self.stage_13(index_points(b1_f, b3_idx))
#             temp23_points = self.stage_23(index_points(b2_f, b3_idx))
#             branch3_points_FP = self.lrelu(b3_f + temp13_points + temp23_points)
#
#             branch4_points_FP = 0
#
#         else:
#             # temp41_points = self.stage_fuse41(b1_xyz, b4_xyz, b1_f, b4_f)
#             # temp31_points = self.stage_fuse31(b1_xyz, b3_xyz, b1_f, b3_f)
#             # temp21_points = self.stage_fuse21(b1_xyz, b2_xyz, b1_f, b2_f)
#             temp41_points = self.stage_fuse41(b1_xyz, b1_f, b4_f, b4_idx)    #######
#             temp31_points = self.stage_fuse31(b1_xyz, b1_f, b3_f, b3_idx)    #######
#             temp21_points = self.stage_fuse21(b1_xyz, b1_f, b2_f, b2_idx)    #######
#             branch1_points = self.lrelu(b1_f + temp21_points + temp31_points + temp41_points)
#
#             # temp42_points = self.stage_fuse42(b2_xyz, b4_xyz, b2_f, b4_f)
#             # temp32_points = self.stage_fuse32(b2_xyz, b3_xyz, b2_f, b3_f)
#             temp42_points = self.stage_fuse42(b2_xyz, b2_f, b4_f, b4_idx)   ######
#             temp32_points = self.stage_fuse32(b2_xyz, b2_f, b3_f, b3_idx)   ######
#             temp12_points = self.stage_12(index_points(b1_f, b2_idx))
#             branch2_points_FP = self.lrelu(b2_f + temp12_points + temp32_points + temp42_points)
#
#             # temp43_point = self.stage_fuse43(b3_xyz, b4_xyz, b3_f, b4_f)
#             temp43_point = self.stage_fuse43(b3_xyz, b3_f, b4_f, b4_idx)   ######
#             temp13_points = self.stage_13(index_points(b1_f, b3_idx))
#             temp23_points = self.stage_23(index_points(b2_f, b3_idx))
#             branch3_points_FP = self.lrelu(b3_f + temp13_points + temp23_points + temp43_point)
#
#             temp14_points = self.stage_14(index_points(b1_f, b4_idx))
#             temp24_points = self.stage_24(index_points(b2_f, b4_idx))
#             temp34_points = self.stage_34(index_points(b3_f, b4_idx))
#             branch4_points_FP = self.lrelu(b4_f + temp14_points + temp24_points + temp34_points)
#
#         # branch4_points_random1 = self.stage_14(
#         #     index_points(b1_f.permute(0, 2, 1).contiguous(), b4_idx).permute(0, 2, 1).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         # branch4_points_random2 = self.stage_24(
#         #     index_points(b2_f.permute(0, 2, 1).contiguous(), b4_idx).permute(0, 2, 1).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         # branch4_points_random3 = self.stage_34(
#         #     index_points(b3_f.permute(0, 2, 1).contiguous(), b4_idx).permute(0, 2, 1).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         # branch4_points_FP = self.stage_conv4(
#         #     (b4_f + branch4_points_random1 + branch4_points_random2 + branch4_points_random3).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         #
#         # # temp43_point_stage4 = self.stage_fuse43(xyz, xyz, b3_f, b4_f)
#         # temp43_point_stage4 = self.stage_fuse43(b3_xyz, b4_xyz, b3_f, b4_f)
#         # branch3_points_random1 = self.stage_13(
#         #     index_points(b1_f.permute(0, 2, 1).contiguous(), b3_idx).permute(0, 2, 1).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         # branch3_points_random2 = self.stage_23(
#         #     index_points(b2_f.permute(0, 2, 1).contiguous(), b3_idx).permute(0, 2, 1).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         # branch3_points_FP = self.stage_conv3(
#         #     (b3_f + branch3_points_random1 + branch3_points_random2 + temp43_point_stage4).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         #
#         # # temp42_points_stage4 = self.stage_fuse42(xyz, xyz, b2_f, b4_f)
#         # # temp32_points_stage4 = self.stage_fuse32(xyz, xyz, b2_f, b3_f)
#         # temp42_points_stage4 = self.stage_fuse42(b2_xyz, b4_xyz, b2_f, b4_f)
#         # temp32_points_stage4 = self.stage_fuse32(b2_xyz, b3_xyz, b2_f, b3_f)
#         # branch2_points_random = self.stage_12(
#         #     index_points(b1_f.permute(0, 2, 1).contiguous(), b2_idx).permute(0, 2, 1).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         # branch2_points_FP = self.stage_conv2(
#         #     (b2_f + branch2_points_random + temp32_points_stage4 + temp42_points_stage4).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#         #
#         # # temp41_points_stage4 = self.stage_fuse41(xyz, xyz, b1_f, b4_f)
#         # # temp31_points_stage4 = self.stage_fuse31(xyz, xyz, b1_f, b3_f)
#         # # temp21_points_stage4 = self.stage_fuse21(xyz, xyz, b1_f, b2_f)
#         # temp41_points_stage4 = self.stage_fuse41(b1_xyz, b4_xyz, b1_f, b4_f)
#         # temp31_points_stage4 = self.stage_fuse31(b1_xyz, b3_xyz, b1_f, b3_f)
#         # temp21_points_stage4 = self.stage_fuse21(b1_xyz, b2_xyz, b1_f, b2_f)
#         # branch1_points = self.stage_conv1(
#         #     (b1_f + temp21_points_stage4 + temp31_points_stage4 + temp41_points_stage4).unsqueeze(
#         #         -1).contiguous()).squeeze(-1)
#
#         return branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP


# class Diffusion_Block(nn.Module):
#     def __init__(self, in_dim, out_dim, num_layer):
#         super(Diffusion_Block, self).__init__()
#         self.lin = nn.Linear(in_dim, out_dim)
#         self.bn = torch.nn.BatchNorm1d(out_dim)
#         self.num_layer = num_layer
#         assert in_dim == out_dim
#
#     def forward(self, g):
#         for _ in range(self.num_layer):
#             # diff
#             g.apply_edges(dgl_fn.u_sub_v("h", "h", "diff_e"))
#             g.edata["diff_e"] = g.edata["e"] * self.lin(g.edata["diff_e"])
#             g.update_all(dgl_fn.copy_e("e", "_"), dgl_fn.sum("_", "bse_e"))
#             g.update_all(dgl_fn.copy_e("diff_e", "m"), dgl_fn.sum("m", "z"))
#             z = g.ndata["z"] / g.ndata["bse_e"].clamp(min=1)
#             # combination
#             z = self.bn(torch.relu(z))
#             g.ndata["h"] = g.ndata["h"] + z
#         return g.ndata["h"]
#
# class PC_Diffuse(nn.Module):
#     def __init__(self, node_size, in_dim, out_dim, num_layer=4):
#         super(PC_Diffuse, self).__init__()
#         # self.batch_size = batch_size
#         self.node_size = node_size
#         self.diffusion_block = Diffusion_Block(in_dim, out_dim, num_layer)
#
#     def forward(self, batch_adj, batch_x):
#         # build dgl-graph
#         index = torch.nonzero(batch_adj).T
#         edge_index = (index[0] * self.node_size) + index[1:]
#         edge_weights = batch_adj[index[0], index[1], index[2]]
#         g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=batch_adj.shape[0] * self.node_size)
#         g.edata["e"] = edge_weights.unsqueeze(dim=-1)
#         g.ndata["h"] = torch.cat(list(batch_x), dim=0)
#         # diffusion
#         z = self.diffusion_block(g)
#         return z.reshape(batch_adj.shape[0], self.node_size, -1)
#
# class Diffusion(nn.Module):
#     def __init__(self, in_channel, out_channel, k_num, adj_shape):
#         super(Diffusion, self).__init__()
#
#         self.adj_shape = adj_shape
#         # self.wgnn = WGNN(adj_shape)
#         self.pc_diffuse = PC_Diffuse(adj_shape, out_channel, out_channel)
#         self.conv = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.k_num = k_num
#
#     def forward(self, xyz, base_xyz, FPS_idx):
#
#
#
#         _, idx = knn_point(self.k_num, base_xyz, xyz)   #####################################
#         output = F.one_hot(idx, num_classes=self.adj_shape).sum(dim=-2)  ################################
#         node_anchor_adj = output.permute(0,2,1).float()                        #########################
#         node_norm = node_anchor_adj/torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=1e-10)   ################
#         anchor_norm = node_anchor_adj/torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=1e-10)   ############
#         adj = torch.matmul(anchor_norm, node_norm.transpose(-1,-2))    ######################
#
#         # new_xyz = index_points2(base_xyz, FPS_idx, xyz)   ##################################
#         new_xyz = mod_index(base_xyz, FPS_idx, xyz)
#
#         x = self.conv(new_xyz.permute(0,2,1).unsqueeze(-1)).squeeze(-1).permute(0,2,1)
#         output = self.pc_diffuse(adj, x)
#         # output = self.wgnn(x, adj)
#
#         return output

#
# class KeepHighResolutionModule(nn.Module):
#
#     def __init__(self, data_C, b1_C, b2_C, b3_C, b4_C):
#         super(KeepHighResolutionModule, self).__init__()
#
#         self.local_num_neighbors = [16, 32]
#         self.neighbour = 16
#
#         self.drop = nn.Dropout(0.5)  ########################
#
#         # self.start1 = SharedMLP(3, b1_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.start2 = SharedMLP(3, b2_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.start3 = SharedMLP(3, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.start4 = SharedMLP(3, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # branch1_offset = torch.zeros([20, 3, 256], dtype=torch.float32)
#         # branch2_offset = torch.zeros([20, 3, 128], dtype=torch.float32)
#         # branch3_offset = torch.zeros([20, 3, 64], dtype=torch.float32)
#         # branch4_offset = torch.zeros([20, 3, 32], dtype=torch.float32)
#
#         # self.offset1 = nn.Parameter(branch1_offset)
#         # self.offset2 = nn.Parameter(branch2_offset)
#         # self.offset3 = nn.Parameter(branch3_offset)
#         # self.offset4 = nn.Parameter(branch4_offset)
#
#         self.conv0 = SharedMLP2(3, 64, first=True)
#         self.Trans0 = LocalTrans(64, 64, 16, usetanh=False, anchorV=False, bn=False)
#         self.Trans1 = LocalTrans(64, 64, 16, usetanh=False, anchorV=False, bn=False)
#
#         # self.Trans12 = LocalTrans(64, 128, 16, usetanh=True, residual=True, anchorV=True)
#         # self.Trans23 = LocalTrans(128, 256, 16, usetanh=True, residual=True, anchorV=True)
#         # self.Trans34 = LocalTrans(256, 512, 16, usetanh=True, residual=True, anchorV=True)
#         # self.Trans_la3 = LocalTrans(3, 12, 16, usetanh=False)
#         # self.Trans_la4 = LocalTrans(3, 12, 16, usetanh=False)
#
#         # self.FPS_offset1 = offset(64, 3, 512)
#         # self.FPS_offset2 = offset(64, 3, 256)
#         # self.FPS_offset3 = offset(64, 3, 128)
#         # self.FPS_offset4 = offset(128, 3, 64)
#         # self.FPS_offset5 = offset(256, 3, 32)
#
#         # self.diffusion1 = Diffusion(3, 16, 16, 1024)
#         # self.diffusion2 = Diffusion(3, 16, 16, 512)
#         # self.diffusion3 = Diffusion(3, 16, 16, 256)
#         # self.diffusion4 = Diffusion(3, 16, 16, 128)
#         # self.diffusion5 = Diffusion(3, 16, 16, 64)
#
#         # self.pool0 = IndexSelect(64, 16)
#         # self.pool1 = IndexSelect(64, 16)
#
#
#
#         self.Trans_pool2 = LocalTrans(64, 128, 16, usetanh=False, residual=True, anchorV=False, bn=False)
#         self.pool2 = IndexSelect(128, 16)
#
#         self.Trans_pool3 = LocalTrans(128, 256, 16, usetanh=False, residual=True, anchorV=False, bn=False)
#         self.pool3 = IndexSelect(256, 16)
#
#         self.Trans_pool4 = LocalTrans(256, 512, 16, usetanh=False, residual=True, anchorV=False, bn=False)
#         self.pool4 = IndexSelect(512, 16)
#
#         # self.conv_up2 = Linear(64, 128, bn=False)
#         # self.conv_up3 = Linear(128, 256, bn=False)
#         # self.conv_up4 = Linear(256, 512, bn=False)
#
#         ###start
#         self.start1 = LocalAggregation(3 * 2 + 7, b1_C, [self.neighbour, ], 3)
#         self.start2 = LocalAggregation(3 * 2 + 7, b2_C, [self.neighbour, ], 3)
#         self.start3 = LocalAggregation(3 * 2 + 7, b3_C, [self.neighbour, ], 3)
#         self.start4 = LocalAggregation(3 * 2 + 7, b4_C, [self.neighbour, ], 3)
#
#         # self.start2 = LocalAggregation(b1_C, b1_C, [self.neighbour, ], 3)
#         # self.start3 = LocalAggregation(b1_C, b1_C, [self.neighbour, ], 3)
#         # self.start4 = LocalAggregation(b1_C, b1_C, [self.neighbour, ], 3)
#
#         ###stage2
#         # self.stage2_trans = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.Trans_stage2_la11 = LocalTrans(64, 64, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage2_la12 = LocalTrans(64, 64, 16, usetanh=True)
#
#         self.Trans_stage2_la21 = LocalTrans(128, 128, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage2_la22 = LocalTrans(128, 128, 16, usetanh=True)
#
#         self.stage2_fuse = Fuse_Stage(64, 128, 256, 512)
#
#         ###stage3
#         # self.stage3_trans = SharedMLP(128, 256, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.Trans_stage3_la11 = LocalTrans(64, 64, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage3_la12 = LocalTrans(64, 64, 16, usetanh=True)
#
#         self.Trans_stage3_la21 = LocalTrans(128, 128, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage3_la22 = LocalTrans(128, 128, 16, usetanh=True)
#
#         self.Trans_stage3_la31 = LocalTrans(256, 256, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage3_la32 = LocalTrans(256, 256, 16, usetanh=True)
#
#         self.stage3_fuse = Fuse_Stage(64, 128, 256, 512)
#
#         ###stage4
#         # self.stage4_trans = SharedMLP(256, 512, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.Trans_stage4_la11 = LocalTrans(64, 64, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage4_la12 = LocalTrans(64, 64, 16, usetanh=True)
#
#         self.Trans_stage4_la21 = LocalTrans(128, 128, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage4_la22 = LocalTrans(128, 128, 16, usetanh=True)
#
#         self.Trans_stage4_la31 = LocalTrans(256, 256, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage4_la32 = LocalTrans(256, 256, 16, usetanh=True)
#
#         self.Trans_stage4_la41 = LocalTrans(512, 512, 16, usetanh=True, anchorV=True)
#         # self.Trans_stage4_la42 = LocalTrans(512, 512, 16, usetanh=True)
#
#         self.stage4_fuse = Fuse_Stage(64, 128, 256, 512)
#
#         # self.start_la1 = LocalAggregation(3 * 2 + 7, b1_C, [self.neighbour, ], 3)
#         # self.start_la2 = LocalAggregation(3 * 2 + 7, b2_C, [self.neighbour, ], 3)
#         # self.start_la3 = LocalAggregation(3 * 2 + 7, b3_C, [self.neighbour, ], 3)
#         # self.start_la4 = LocalAggregation(3 * 2 + 7, b4_C, [self.neighbour, ], 3)
#         #
#         #
#         # # self.start_localGraph1 = GetGraph(3, b1_C, 19, residual=True)
#         # # self.start_localGraph2 = GetGraph(3, b2_C, 19, residual=True)
#         # # self.start_localGraph3 = GetGraph(3, b3_C, 19, residual=True)
#         # # self.start_localGraph4 = GetGraph(3, b4_C, 19, residual=True)
#         #
#         # self.globalTrans_start_la1 = LocalTrans(128, 128, 80, stage=False, usetanh=True)
#         # self.globalTrans_start_la2 = LocalTrans(64, 64, 80, stage=False, usetanh=True)  #############
#         # self.globalTrans_start_la3 = LocalTrans(32, 32, 80, stage=False, usetanh=True)
#         # self.globalTrans_start_la4 = LocalTrans(16, 16, 80, stage=False, usetanh=True)  ###############
#         #
#         # self.localTrans_start_la1 = LocalTrans(128, 128, 80, stage=False, usetanh=True)
#         # self.localTrans_start_la2 = LocalTrans(64, 64, 80, stage=True, usetanh=True)  #############
#         # self.localTrans_start_la3 = LocalTrans(32, 32, 80, stage=True, usetanh=True)
#         # self.localTrans_start_la4 = LocalTrans(16, 16, 80, stage=True, usetanh=True)  ###############
#         #
#         # self.start1 = SharedMLP(256, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.start2 = SharedMLP(128, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.start3 = SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.start4 = SharedMLP(32, 16, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         #
#         # self.start_fuse = Fuse_Stage(128, 64, 32, 16)
#         #
#         #
#         #
#         #
#         # ###stage2
#         # # self.stage2_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
#         # # self.stage2_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
#         # # self.stage2_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
#         # # self.stage2_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])
#         #
#         # # self.stage2_localGraph1 = GetGraph(b1_C, b1_C, 19)
#         # # self.stage2_localGraph2 = GetGraph(b2_C, b2_C, 19)
#         # # self.stage2_localGraph3 = GetGraph(b3_C, b3_C, 19)
#         # # self.stage2_localGraph4 = GetGraph(b4_C, b4_C, 19)
#         #
#         #
#         #
#         # # self.localTrans_stage2_xyz_la1 = LocalTransXyz(3, 64, 80, stage=True)
#         # # self.localTrans_stage2_xyz_la2 = LocalTransXyz(3, 64, 80, stage=True)  #############
#         # # self.localTrans_stage2_xyz_la3 = LocalTransXyz(3, 64, 80, stage=True)
#         # # self.localTrans_stage2_xyz_la4 = LocalTransXyz(3, 64, 80, stage=True)  ###############
#         #
#         #
#         # self.globalTrans_stage2_la1 = LocalTrans(128, 128, 80, stage=False, usetanh=True)
#         # self.globalTrans_stage2_la2 = LocalTrans(64, 64, 80, stage=False, usetanh=True)  #############
#         # self.globalTrans_stage2_la3 = LocalTrans(32, 32, 80, stage=False, usetanh=True)
#         # self.globalTrans_stage2_la4 = LocalTrans(16, 16, 80, stage=False, usetanh=True)  ###############
#         #
#         # self.localTrans_stage2_la1 = LocalTrans(128, 128, 80, stage=False, usetanh=True)
#         # self.localTrans_stage2_la2 = LocalTrans(64, 64, 80, stage=True, usetanh=True)  #############
#         # self.localTrans_stage2_la3 = LocalTrans(32, 32, 80, stage=True, usetanh=True)
#         # self.localTrans_stage2_la4 = LocalTrans(16, 16, 80, stage=True, usetanh=True)  ###############
#         #
#         # self.stage21 = SharedMLP(256, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage22 = SharedMLP(128, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage23 = SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage24 = SharedMLP(32, 16, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         #
#         # # self.stage2_fuse_atten = FuseAtten(128, 64)
#         #
#         # self.stage2_fuse = Fuse_Stage(128, 64, 32, 16)
#         #
#         #
#         #
#         #
#         #
#         # ###stage3
#         # # self.stage3_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
#         # # self.stage3_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
#         # # self.stage3_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
#         # # self.stage3_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])
#         #
#         # # self.stage3_localGraph1 = GetGraph(b1_C, b1_C, 19)
#         # # self.stage3_localGraph2 = GetGraph(b2_C, b2_C, 19)
#         # # self.stage3_localGraph3 = GetGraph(b3_C, b3_C, 19)
#         # # self.stage3_localGraph4 = GetGraph(b4_C, b4_C, 19)
#         #
#         #
#         # # self.localTrans_stage3_xyz_la1 = LocalTransXyz(3, 64, 80, stage=True)
#         # # self.localTrans_stage3_xyz_la2 = LocalTransXyz(3, 64, 80, stage=True)  #############
#         # # self.localTrans_stage3_xyz_la3 = LocalTransXyz(3, 64, 80, stage=True)
#         # # self.localTrans_stage3_xyz_la4 = LocalTransXyz(3, 64, 80, stage=True)  ###############
#         #
#         # self.globalTrans_stage3_la1 = LocalTrans(128, 128, 80, stage=False, usetanh=True)
#         # self.globalTrans_stage3_la2 = LocalTrans(64, 64, 80, stage=False, usetanh=True)  #############
#         # self.globalTrans_stage3_la3 = LocalTrans(32, 32, 80, stage=False, usetanh=True)
#         # self.globalTrans_stage3_la4 = LocalTrans(16, 16, 80, stage=False, usetanh=True)  ###############
#         #
#         #
#         # self.localTrans_stage3_la1 = LocalTrans(128, 128, 80, stage=False, usetanh=True)
#         # self.localTrans_stage3_la2 = LocalTrans(64, 64, 80, stage=True, usetanh=True)  #############
#         # self.localTrans_stage3_la3 = LocalTrans(32, 32, 80, stage=True, usetanh=True)
#         # self.localTrans_stage3_la4 = LocalTrans(16, 16, 80, stage=True, usetanh=True)  ###############
#         #
#         #
#         # self.stage31 = SharedMLP(256, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage32 = SharedMLP(128, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage33 = SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage34 = SharedMLP(32, 16, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         #
#         #
#         # # self.stage3_fuse_atten = FuseAtten(128, 64)
#         #
#         # self.stage3_fuse = Fuse_Stage(128, 64, 32, 16)
#         #
#         #
#         #
#         #
#         #
#         # ###stage4
#         # # self.stage4_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
#         # # self.stage4_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
#         # # self.stage4_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
#         # # self.stage4_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])
#         #
#         # # self.stage4_localGraph1 = GetGraph(b1_C, b1_C, 19)
#         # # self.stage4_localGraph2 = GetGraph(b2_C, b2_C, 19)
#         # # self.stage4_localGraph3 = GetGraph(b3_C, b3_C, 19)
#         # # self.stage4_localGraph4 = GetGraph(b4_C, b4_C, 19)
#         #
#         #
#         # # self.localTrans_stage4_xyz_la1 = LocalTransXyz(3, 64, 80, stage=True)
#         # # self.localTrans_stage4_xyz_la2 = LocalTransXyz(3, 64, 80, stage=True)  #############
#         # # self.localTrans_stage4_xyz_la3 = LocalTransXyz(3, 64, 80, stage=True)
#         # # self.localTrans_stage4_xyz_la4 = LocalTransXyz(3, 64, 80, stage=True)  ###############
#         #
#         # self.globalTrans_stage4_la1 = LocalTrans(128, 128, 80, stage=False, usetanh=True)
#         # self.globalTrans_stage4_la2 = LocalTrans(64, 64, 80, stage=False, usetanh=True)  #############
#         # self.globalTrans_stage4_la3 = LocalTrans(32, 32, 80, stage=False, usetanh=True)
#         # self.globalTrans_stage4_la4 = LocalTrans(16, 16, 80, stage=False, usetanh=True)  ###############
#         #
#         # self.localTrans_stage4_la1 = LocalTrans(128, 128, 80, stage=False, usetanh=True)
#         # self.localTrans_stage4_la2 = LocalTrans(64, 64, 80, stage=True, usetanh=True)  #############
#         # self.localTrans_stage4_la3 = LocalTrans(32, 32, 80, stage=True, usetanh=True)
#         # self.localTrans_stage4_la4 = LocalTrans(16, 16, 80, stage=True, usetanh=True)  ###############c
#         #
#         #
#         # self.stage41 = SharedMLP(256, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage42 = SharedMLP(128, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage43 = SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.stage44 = SharedMLP(32, 16, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         #
#         # # self.stage4_fuse_atten = FuseAtten(128, 64)
#         #
#         # self.stage4_fuse = Fuse_Stage(128, 64, 32, 16)
#
#         # self.localTrans_stage5_la1 = LocalTrans(64, 64, 16)
#         # self.localTrans_stage5_la2 = LocalTrans(64, 64, 16)  #############
#         # self.localTrans_stage5_la3 = LocalTrans(64, 64, 16)
#         # self.localTrans_stage5_la4 = LocalTrans(64, 64, 16)  ###############
#         #
#         # self.stage5_fuse = Fuse_Stage(64, 64, 64, 64)
#         #
#         #
#         # self.localTrans_stage6_la1 = LocalTrans(64, 64, 16)
#         # self.localTrans_stage6_la2 = LocalTrans(64, 64, 16)  #############
#         # self.localTrans_stage6_la3 = LocalTrans(64, 64, 16)
#         # self.localTrans_stage6_la4 = LocalTrans(64, 64, 16)  ###############
#         #
#         # self.stage6_fuse = Fuse_Stage(64, 64, 64, 64)
#         #
#         #
#         # self.localTrans_stage7_la1 = LocalTrans(64, 64, 16)
#         # self.localTrans_stage7_la2 = LocalTrans(64, 64, 16)  #############
#         # self.localTrans_stage7_la3 = LocalTrans(64, 64, 16)
#         # self.localTrans_stage7_la4 = LocalTrans(64, 64, 16)  ###############
#         #
#         # self.stage7_fuse = Fuse_Stage(64, 64, 64, 64)
#         #
#         #
#         # self.localTrans_stage8_la1 = LocalTrans(64, 64, 16)
#         # self.localTrans_stage8_la2 = LocalTrans(64, 64, 16)  #############
#         # self.localTrans_stage8_la3 = LocalTrans(64, 64, 16)
#         # self.localTrans_stage8_la4 = LocalTrans(64, 64, 16)  ###############
#         #
#         # self.stage8_fuse = Fuse_Stage(64, 64, 64, 64)
#
#         ###final stage
#         # self.convFinal = conv1d(b1_C, b1_C)
#
#         self.conv_x1 = Linear(64, 128, bn=False)
#         self.conv_x2 = Linear(64, 128, bn=False)
#         self.conv_x3 = Linear(64, 128, bn=False)
#         self.conv_x4 = Linear(64, 128, bn=False)
#
#         # self.conv_x1 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x2 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x3 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x4 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # self.conv_x5 = SharedMLP(64, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x6 = SharedMLP(64, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x7 = SharedMLP(64, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x8 = SharedMLP(64, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # self.head1 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.head12 = SharedMLP(128, 256, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         #
#         # self.head2 = SharedMLP(128, 256, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.head22 = SharedMLP(256, 512, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         #
#         # self.head3 = SharedMLP(256, 512, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.head32 = SharedMLP(512, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # #
#         # self.head4 = SharedMLP(512, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         # self.final = SharedMLP(512, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.final_class = SharedMLP(2048, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.final = Linear(512, 1024, bn=False)
#         self.final_class = nn.Linear(2048, 1024)
#         self.bn = nn.BatchNorm1d(1024)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2)
#
#     def forward(self, xyz, points):
#
#         if points is not None:
#             branch1_norm = points
#         else:
#             branch1_norm = None
#
#         # points = self.conv0(xyz, xyz)
#         points = self.start1(features=xyz, norm=branch1_norm)
#
#         # value0, idx0 = self.pool0(xyz.permute(0,2,1), points)
#         # FPS_xyz0 = index_points(xyz.permute(0,2,1), idx0)
#
#         FPS_idx0 = farthest_point_sample(xyz.permute(0, 2, 1).contiguous(), 512)
#         FPS_xyz0 = index_points(xyz.permute(0, 2, 1).contiguous(), FPS_idx0)
#
#         # FPS_xyz0, _ = self.FPS_offset1(xyz.permute(0, 2, 1), points, FPS_idx0, FPS_xyz0_gt)
#         # FPS_xyz0, FPS_idx0 = self.FPS_offset1(xyz.permute(0,2,1), points)
#         # d1 = self.diffusion1(FPS_xyz0, xyz.permute(0,2,1), FPS_idx0)
#         # d1_idx = FPS_idx0
#
#         points0 = self.Trans0(xyz=FPS_xyz0, base_xyz=xyz.permute(0, 2, 1).contiguous(), features=points,
#                               FPS_idx=FPS_idx0)
#         # points0_f = self.start2(features = FPS_xyz0.permute(0,2,1), norm = None)
#         # points0 = points0 + points0_f
#
#         # value1, idx1 = self.pool1(FPS_xyz0, points0)
#         # FPS_xyz1 = index_points(FPS_xyz0, idx1)
#
#         FPS_idx1 = farthest_point_sample(FPS_xyz0, 256)
#         FPS_xyz1 = index_points(FPS_xyz0, FPS_idx1)
#
#         # FPS_xyz1, _ = self.FPS_offset2(FPS_xyz0_gt, points0, FPS_idx1, FPS_xyz1_gt)
#         # FPS_xyz1, FPS_idx1 = self.FPS_offset2(FPS_xyz0, points0)
#         # d2 = self.diffusion2(FPS_xyz1, FPS_xyz0, FPS_idx1)
#         # d2_idx = FPS_idx1
#
#         point1 = self.Trans1(xyz=FPS_xyz1, base_xyz=FPS_xyz0, features=points0, FPS_idx=FPS_idx1)
#         # point1_f = self.start3(features = FPS_xyz1.permute(0,2,1), norm = None)
#         # point1 = point1 + point1_f
#
#         ###stage1
#         # points = self.start1(features=branch1_xyz, norm=branch1_norm)
#         # branch1_xyz_offset = branch1_xyz + self.offset1(branch1_xyz)
#         # branch1_points = self.Trans_la1(features=None, xyz=branch1_xyz_offset, base_xyz=xyz)
#         # branch1_points = self.Trans_la1(features=None, xyz=branch1_xyz, base_xyz=xyz)
#         # branch1_points = self.start1(features=branch1_points, norm=None)
#         branch1_xyz = FPS_xyz1
#         branch1_points = point1
#         # branch1_idx = FPS_idx1
#         # points = self.start1(features=xyz, norm=branch1_norm)
#         #
#         # branch1_idx = farthest_point_sample(xyz.permute(0,2,1).contiguous(), 512)
#         # branch1_xyz = index_points(xyz.permute(0, 2, 1).contiguous(), branch1_idx)
#         # points = index_points(points.permute(0, 2, 1).contiguous(), branch1_idx)
#         #
#         # points = points.permute(0,2,1) + self.start2(features=points.permute(0,2,1), xyz=branch1_xyz, norm=branch1_norm)
#         #
#         # branch1_idx = farthest_point_sample(branch1_xyz, 256)
#         # branch1_xyz = index_points(branch1_xyz, branch1_idx)
#         # points = index_points(points.permute(0, 2, 1).contiguous(), branch1_idx)
#         #
#         # branch1_xyz = branch1_xyz.permute(0,2,1)    ##### become  (B,C,N)
#         # points = points.permute(0,2,1)              ##### become  (B,C,N)
#
#         # points = points + self.start2(features=points, xyz=branch1_xyz.permute(0,2,1), norm=branch1_norm)
#         # points = points + self.start3(features=points, xyz=branch1_xyz.permute(0,2,1), norm=branch1_norm)
#         # points = points + self.start4(features=points, xyz=branch1_xyz.permute(0,2,1), norm=branch1_norm)
#
#         x1 = self.conv_x1(branch1_points)
#
#         ###stage2
#         # branch1_points_temp = self.conv_up2(branch1_points)
#         branch2_points_FP = self.Trans_pool2(xyz=branch1_xyz, base_xyz=branch1_xyz, features=branch1_points)
#         ret2, branch2_xyz, branch2_points_FP, idx2 = self.pool2(branch1_xyz, branch2_points_FP)
#
#         # branch2_xyz = index_points(branch1_xyz, idx2)
#
#         # branch2_idx_FP = farthest_point_sample(branch1_xyz, 128)
#         # branch2_xyz = index_points(FPS_xyz1, branch2_idx_FP)
#
#         # branch2_xyz, _ = self.FPS_offset3(FPS_xyz1_gt, branch1_points, branch2_idx_FP, branch2_xyz_gt)
#         # branch2_xyz, branch2_idx_FP = self.FPS_offset3(branch1_xyz, branch1_points)
#         # d3 = self.diffusion3(branch2_xyz, branch1_xyz, branch2_idx_FP)
#         # d3_idx = branch2_idx_FP
#
#         # branch2_points_FP = self.Trans12(xyz=branch2_xyz.permute(0,2,1), base_xyz=branch1_xyz.permute(0, 2, 1), features=branch1_points, FPS_idx=idx2, value=value2)
#
#         # branch2_points_FP_f = self.start2(features = branch2_xyz.permute(0,2,1), norm = None)
#         # branch2_points_FP = branch2_points_FP + branch2_points_FP_f
#
#         # branch2_xyz_offset = branch2_xyz + self.offset2(branch2_xyz)
#         # branch2_points = self.Trans_la2(features=None, xyz=branch2_xyz_offset, base_xyz=branch1_xyz_offset)
#         # branch2_points = self.Trans_la2(features=None, xyz=branch2_xyz, base_xyz=branch1_xyz)
#         # branch2_points = self.start2(features=branch2_points, norm=None)
#         #
#         # branch2_points_FP = index_points(branch1_points.permute(0, 2, 1).contiguous(), branch2_idx_FP).permute(0,2,1)
#         # branch2_points_FP = self.stage2_trans(branch2_points_FP.unsqueeze(-1)).squeeze(-1)
#         #
#         # branch2_points_FP = branch2_points_FP + branch2_points
#
#         branch1_points = self.Trans_stage2_la11(xyz=branch1_xyz, base_xyz=branch1_xyz, features=branch1_points)
#         # branch1_points = self.Trans_stage2_la12(branch1_points)
#
#         branch2_points_FP = self.Trans_stage2_la21(xyz=branch2_xyz, base_xyz=branch2_xyz, features=branch2_points_FP)
#         # branch2_points_FP = self.Trans_stage2_la22(branch2_points_FP)
#
#         # branch1_points_temp = self.Trans_stage2_la11(branch1_points)
#         # branch1_points = branch1_points + self.Trans_stage2_la12(branch1_points_temp)
#         #
#         # branch2_points_FP_temp = self.Trans_stage2_la21(branch2_points_FP)
#         # branch2_points_FP = branch2_points_FP + self.Trans_stage2_la22(branch2_points_FP_temp)
#
#         branch1_points, branch2_points_FP, _, _ = self.stage2_fuse(b1_f=branch1_points, b2_f=branch2_points_FP,
#                                                                    b2_idx=idx2,
#                                                                    b1_xyz=branch1_xyz, b2_xyz=branch2_xyz)
#
#         x2 = self.conv_x2(branch1_points)
#
#         ###stage3
#         # branch2_points_FP_temp = self.conv_up3(branch2_points_FP)
#         branch3_points_FP = self.Trans_pool3(xyz=branch2_xyz, base_xyz=branch2_xyz, features=branch2_points_FP)
#         ret3, branch3_xyz, branch3_points_FP, idx3 = self.pool3(branch2_xyz, branch3_points_FP)
#
#         # branch3_xyz = index_points(branch2_xyz, idx3)
#
#         # branch3_idx_FP = farthest_point_sample(branch2_xyz, 64)
#         # branch3_xyz = index_points(branch2_xyz, branch3_idx_FP)
#
#         # branch3_xyz, _ = self.FPS_offset4(branch2_xyz_gt, branch2_points_FP, branch3_idx_FP, branch3_xyz_gt)
#         # branch3_xyz, branch3_idx_FP = self.FPS_offset4(branch2_xyz, branch2_points_FP)
#         # d4 = self.diffusion4(branch3_xyz, branch2_xyz, branch3_idx_FP)
#         # d4_idx = branch3_idx_FP
#
#         # branch3_points_FP = self.Trans23(xyz=branch3_xyz.permute(0,2,1), base_xyz=branch2_xyz.permute(0, 2, 1), features=branch2_points_FP, FPS_idx=idx3, value=value3)
#
#         # branch3_points_FP_f = self.start3(features = branch3_xyz.permute(0,2,1), norm = None)
#         # branch3_points_FP = branch3_points_FP + branch3_points_FP_f
#
#         # branch3_xyz_offset = branch3_xyz + self.offset3(branch3_xyz)
#         # branch3_points = self.Trans_la3(features=None, xyz=branch3_xyz_offset, base_xyz=branch2_xyz_offset)
#         # branch3_points = self.Trans_la3(features=None, xyz=branch3_xyz, base_xyz=branch2_xyz)
#         # branch3_points = self.start3(features=branch3_points, norm=None)
#         #
#         # branch3_points_FP = index_points(branch2_points_FP.permute(0, 2, 1).contiguous(), branch3_idx_FP).permute(0,2,1)
#         # branch3_points_FP = self.stage3_trans(branch3_points_FP.unsqueeze(-1)).squeeze(-1)
#         #
#         # branch3_points_FP = branch3_points_FP + branch3_points
#
#         branch1_points = self.Trans_stage3_la11(xyz=branch1_xyz, base_xyz=branch1_xyz, features=branch1_points)
#         # branch1_points = self.Trans_stage3_la12(branch1_points)
#
#         branch2_points_FP = self.Trans_stage3_la21(xyz=branch2_xyz, base_xyz=branch2_xyz, features=branch2_points_FP)
#         # branch2_points_FP = self.Trans_stage3_la22(branch2_points_FP)
#
#         branch3_points_FP = self.Trans_stage3_la31(xyz=branch3_xyz, base_xyz=branch3_xyz, features=branch3_points_FP)
#         # branch3_points_FP = self.Trans_stage3_la32(branch3_points_FP)
#
#         # branch1_points_temp = self.Trans_stage3_la11(branch1_points)
#         # branch1_points = branch1_points + self.Trans_stage3_la12(branch1_points_temp)
#         #
#         # branch2_points_FP_temp = self.Trans_stage3_la21(branch2_points_FP)
#         # branch2_points_FP = branch2_points_FP + self.Trans_stage3_la22(branch2_points_FP_temp)
#         #
#         # branch3_points_FP_temp = self.Trans_stage3_la31(branch3_points_FP)
#         # branch3_points_FP = branch3_points_FP + self.Trans_stage3_la32(branch3_points_FP_temp)
#
#         branch1_points, branch2_points_FP, branch3_points_FP, _ = self.stage3_fuse(b1_f=branch1_points,
#                                                                                    b2_f=branch2_points_FP,
#                                                                                    b3_f=branch3_points_FP,
#                                                                                    b2_idx=idx2, b3_idx=idx3,
#                                                                                    b1_xyz=branch1_xyz,
#                                                                                    b2_xyz=branch2_xyz,
#                                                                                    b3_xyz=branch3_xyz)
#
#         x3 = self.conv_x3(branch1_points)
#
#         ###stage4
#         # branch3_points_FP_temp = self.conv_up4(branch3_points_FP)
#         branch4_points_FP = self.Trans_pool4(xyz=branch3_xyz, base_xyz=branch3_xyz, features=branch3_points_FP)
#         ret4, branch4_xyz, branch4_points_FP, idx4 = self.pool4(branch3_xyz, branch4_points_FP)
#
#         # branch4_xyz = index_points(branch3_xyz, idx4)
#
#         # branch4_idx_FP = farthest_point_sample(branch3_xyz, 32)
#         # branch4_xyz = index_points(branch3_xyz, branch4_idx_FP)
#
#         # branch4_xyz, _ = self.FPS_offset5(branch3_xyz_gt, branch3_points_FP, branch4_idx_FP, branch4_xyz_gt)
#         # branch4_xyz, branch4_idx_FP = self.FPS_offset5(branch3_xyz, branch3_points_FP)
#         # d5 = self.diffusion5(branch4_xyz, branch3_xyz, branch4_idx_FP)
#         # d5_idx = branch4_idx_FP
#
#         # branch4_points_FP = self.Trans34(xyz=branch4_xyz.permute(0,2,1), base_xyz=branch3_xyz.permute(0, 2, 1), features=branch3_points_FP, FPS_idx=idx4, value=value4)
#
#         # branch4_points_FP_f = self.start4(features = branch4_xyz.permute(0,2,1), norm = None)
#         # branch4_points_FP = branch4_points_FP + branch4_points_FP_f
#
#         # branch4_xyz_offset = branch4_xyz + self.offset4(branch4_xyz)
#         # branch4_points = self.Trans_la4(features=None, xyz=branch4_xyz_offset, base_xyz=branch3_xyz_offset)
#         # branch4_points = self.Trans_la4(features=None, xyz=branch4_xyz, base_xyz=branch3_xyz)
#         # branch4_points = self.start4(features=branch4_points, norm=None)
#         #
#         # branch4_points_FP = index_points(branch3_points_FP.permute(0, 2, 1).contiguous(), branch4_idx_FP).permute(0,2,1)
#         # branch4_points_FP = self.stage4_trans(branch4_points_FP.unsqueeze(-1)).squeeze(-1)
#         #
#         # branch4_points_FP = branch4_points_FP + branch4_points
#
#         branch1_points = self.Trans_stage4_la11(xyz=branch1_xyz, base_xyz=branch1_xyz, features=branch1_points)
#         # branch1_points = self.Trans_stage4_la12(branch1_points)
#
#         branch2_points_FP = self.Trans_stage4_la21(xyz=branch2_xyz, base_xyz=branch2_xyz, features=branch2_points_FP)
#         # branch2_points_FP = self.Trans_stage4_la22(branch2_points_FP)
#
#         branch3_points_FP = self.Trans_stage4_la31(xyz=branch3_xyz, base_xyz=branch3_xyz, features=branch3_points_FP)
#         # branch3_points_FP = self.Trans_stage4_la32(branch3_points_FP)
#
#         branch4_points_FP = self.Trans_stage4_la41(xyz=branch4_xyz, base_xyz=branch4_xyz, features=branch4_points_FP)
#         # branch4_points_FP = self.Trans_stage4_la42(branch4_points_FP)
#
#         # branch1_points_temp = self.Trans_stage4_la11(branch1_points)
#         # branch1_points = branch1_points + self.Trans_stage4_la12(branch1_points_temp)
#         #
#         # branch2_points_FP_temp = self.Trans_stage4_la21(branch2_points_FP)
#         # branch2_points_FP = branch2_points_FP + self.Trans_stage4_la22(branch2_points_FP_temp)
#         #
#         # branch3_points_FP_temp = self.Trans_stage4_la31(branch3_points_FP)
#         # branch3_points_FP = branch3_points_FP + self.Trans_stage4_la32(branch3_points_FP_temp)
#         #
#         # branch4_points_FP_temp = self.Trans_stage4_la41(branch4_points_FP)
#         # branch4_points_FP = branch4_points_FP + self.Trans_stage4_la42(branch4_points_FP_temp)
#
#         branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage3_fuse(b1_f=branch1_points,
#                                                                                                    b2_f=branch2_points_FP,
#                                                                                                    b3_f=branch3_points_FP,
#                                                                                                    b4_f=branch4_points_FP,
#                                                                                                    b2_idx=idx2,
#                                                                                                    b3_idx=idx3,
#                                                                                                    b4_idx=idx4,
#                                                                                                    b1_xyz=branch1_xyz,
#                                                                                                    b2_xyz=branch2_xyz,
#                                                                                                    b3_xyz=branch3_xyz,
#                                                                                                    b4_xyz=branch4_xyz)
#
#         x4 = self.conv_x4(branch1_points)
#
#         # # temp = branch1_xyz.permute(0, 2, 1).contiguous()  ## B N C
#         # # dist, idx1 = knn_point(80, temp, temp)  ### B N K
#         # branch1_points = self.start_la1(branch1_xyz, norm=branch1_norm)
#         # # branch1_points = self.start_localGraph1(branch1_xyz)
#         # branch1_points_global = self.globalTrans_start_la1(branch1_points)
#         # branch1_points_local = self.localTrans_start_la1(branch1_points, branch1_xyz, branch1_xyz, branch1_points)
#         # branch1_points_cat = torch.cat((branch1_points_local, branch1_points_global), dim=1)
#         # branch1_points = self.start1(branch1_points_cat.unsqueeze(-1)).squeeze(-1) + branch1_points
#         #
#         #
#         # branch2_idx_FP = farthest_point_sample(branch1_xyz.permute(0,2,1).contiguous(), 512)
#         # branch2_xyz = index_points(branch1_xyz.permute(0,2,1).contiguous(), branch2_idx_FP)
#         # if points is not None:
#         #     branch2_norm = index_points(branch1_norm.permute(0,2,1), branch2_idx_FP).permute(0,2,1).contiguous()
#         # else:
#         #     branch2_norm = None
#         # branch2_points_FP = self.start_la2(branch2_xyz.permute(0, 2, 1).contiguous(), norm=branch2_norm)  # FPS generate branch2
#         # # branch2_points_FP = self.start_localGraph2(branch2_xyz.permute(0,2,1))
#         # # dist, idx2 = knn_point(80, branch2_xyz, branch2_xyz)  ### B N K
#         # branch2_xyz = branch2_xyz.permute(0,2,1)
#         # branch2_points_FP_global = self.globalTrans_start_la2(branch2_points_FP)
#         # branch2_points_FP_local = self.localTrans_start_la2(branch2_points_FP, branch1_xyz, branch2_xyz, branch1_points, branch2_idx_FP)
#         # branch2_points_FP_cat = torch.cat((branch2_points_FP_local, branch2_points_FP_global), dim=1)
#         # branch2_points_FP = self.start2(branch2_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch2_points_FP
#         #
#         #
#         # branch2_xyz = branch2_xyz.permute(0, 2, 1)
#         # branch3_idx_FP = farthest_point_sample(branch2_xyz, 256)
#         # branch3_xyz = index_points(branch2_xyz, branch3_idx_FP)
#         # if points is not None:
#         #     branch3_norm = index_points(branch2_norm.permute(0,2,1), branch3_idx_FP).permute(0,2,1).contiguous()
#         # else:
#         #     branch3_norm = None
#         # branch3_points_FP = self.start_la3(branch3_xyz.permute(0,2,1).contiguous(), norm=branch3_norm)
#         # # branch3_points_FP = self.start_localGraph3(branch3_xyz.permute(0,2,1))
#         # # dist, idx3 = knn_point(80, branch3_xyz, branch3_xyz)  ### B N K
#         # branch3_xyz = branch3_xyz.permute(0,2,1)
#         # branch2_xyz = branch2_xyz.permute(0, 2, 1)
#         # branch3_points_FP_global = self.globalTrans_start_la3(branch3_points_FP)
#         # branch3_points_FP_local = self.localTrans_start_la3(branch3_points_FP, branch2_xyz, branch3_xyz, branch2_points_FP, branch3_idx_FP)
#         # branch3_points_FP_cat = torch.cat((branch3_points_FP_local, branch3_points_FP_global), dim=1)
#         # branch3_points_FP = self.start3(branch3_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch3_points_FP
#         #
#         # branch3_xyz = branch3_xyz.permute(0, 2, 1)
#         # branch4_idx_FP = farthest_point_sample(branch3_xyz, 128)
#         # branch4_xyz = index_points(branch3_xyz, branch4_idx_FP)
#         # if points is not None:
#         #     branch4_norm = index_points(branch3_norm.permute(0,2,1), branch4_idx_FP).permute(0,2,1).contiguous()
#         # else:
#         #     branch4_norm = None
#         # branch4_points_FP = self.start_la4(branch4_xyz.permute(0,2,1).contiguous(), norm=branch4_norm)
#         # # branch4_points_FP = self.start_localGraph4(branch4_xyz.permute(0,2,1))
#         # # dist, idx4 = knn_point(80, branch4_xyz, branch4_xyz)  ### B N K
#         # branch4_xyz = branch4_xyz.permute(0,2,1)
#         # branch3_xyz = branch3_xyz.permute(0, 2, 1)
#         # branch4_points_FP_global = self.globalTrans_start_la4(branch4_points_FP)
#         # branch4_points_FP_local = self.localTrans_start_la4(branch4_points_FP, branch3_xyz, branch4_xyz, branch3_points_FP, branch4_idx_FP)
#         # branch4_points_FP_cat = torch.cat((branch4_points_FP_local, branch4_points_FP_global), dim=1)
#         # branch4_points_FP = self.start4(branch4_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch4_points_FP
#         #
#         # ################ fuse
#         # branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.start_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
#         #                                                                                      branch2_idx_FP, branch3_idx_FP, branch4_idx_FP, branch1_xyz, branch2_xyz, branch3_xyz, branch4_xyz)
#         #
#         # # x1 = self.conv_x1(branch1_points.unsqueeze(-1).contiguous())
#         # x1 = branch1_points.unsqueeze(-1).contiguous()
#         # # x1 = self.conv_x1(branch4_points_FP.unsqueeze(-1).contiguous())
#         #
#         # # branch1_xyz = branch1_xyz.permute(0,2,1).contiguous()
#         # # branch2_xyz = branch2_xyz.permute(0, 2, 1).contiguous()
#         # # branch3_xyz = branch3_xyz.permute(0, 2, 1).contiguous()
#         # # branch4_xyz = branch4_xyz.permute(0, 2, 1).contiguous()
#         #
#         #
#         #
#         #
#         # ############## Local Aggregation  stage2
#         # # branch1_points = self.stage2_la1(branch1_points, branch1_xyz, norm=branch1_norm)
#         # # branch2_points_FP = self.stage2_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
#         # # branch3_points_FP = self.stage2_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
#         # # branch4_points_FP = self.stage2_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)
#         #
#         #
#         #
#         # # branch1_points = self.stage2_localGraph1(branch1_points)
#         # # branch2_points_FP = self.stage2_localGraph2(branch2_points_FP)
#         #
#         # # branch3_points_FP = self.stage2_localGraph3(branch3_points_FP)
#         # # branch4_points_FP = self.stage2_localGraph4(branch4_points_FP)
#         #
#         #
#         #
#         #
#         #
#         # branch1_points_global = self.globalTrans_stage2_la1(branch1_points)
#         # branch1_points_local = self.localTrans_stage2_la1(branch1_points, branch1_xyz, branch1_xyz, branch1_points)
#         # branch1_points_cat = torch.cat((branch1_points_local, branch1_points_global), dim=1)
#         # branch1_points = self.stage21(branch1_points_cat.unsqueeze(-1)).squeeze(-1) + branch1_points
#         #
#         # # branch2_idx_FP = farthest_point_sample(branch1_xyz, 512)
#         # # branch2_xyz = index_points(branch1_xyz, branch2_idx_FP)
#         #
#         # # branch2_xyz, branch2_idx = random_sample(branch1_xyz, 512)
#         #
#         # branch2_points_FP_global = self.globalTrans_stage2_la2(branch2_points_FP)
#         # branch2_points_FP_local = self.localTrans_stage2_la2(branch2_points_FP, branch1_xyz, branch2_xyz, branch1_points, branch2_idx_FP)
#         # branch2_points_FP_cat = torch.cat((branch2_points_FP_local, branch2_points_FP_global), dim=1)
#         # branch2_points_FP = self.stage22(branch2_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch2_points_FP
#         #
#         # # branch2_points = index_points(branch1_points.permute(0,2,1).contiguous(), branch2_idx_FP).permute(0,2,1)
#         # # branch2_points = self.globalTrans_stage2_la2(branch2_points)
#         # # branch2_points_FP = branch2_points_FP + branch2_points
#         # # branch3_idx_FP = farthest_point_sample(branch2_xyz, 256)
#         # # branch3_xyz = index_points(branch2_xyz, branch3_idx_FP)
#         #
#         # # branch3_xyz, branch3_idx = random_sample(branch2_xyz, 256)
#         #
#         # branch3_points_FP_global = self.globalTrans_stage2_la3(branch3_points_FP)
#         # branch3_points_FP_local = self.localTrans_stage2_la3(branch3_points_FP, branch2_xyz, branch3_xyz, branch2_points_FP, branch3_idx_FP)
#         # branch3_points_FP_cat = torch.cat((branch3_points_FP_local, branch3_points_FP_global), dim=1)
#         # branch3_points_FP = self.stage23(branch3_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch3_points_FP
#         #
#         #
#         # # branch3_points = index_points(branch2_points_FP.permute(0,2,1).contiguous(), branch3_idx_FP).permute(0,2,1)
#         # # branch3_points = self.globalTrans_stage2_la3(branch3_points)
#         # # branch3_points_FP = branch3_points_FP + branch3_points
#         # # branch4_idx_FP = farthest_point_sample(branch3_xyz, 128)
#         # # branch4_xyz = index_points(branch3_xyz, branch4_idx_FP)
#         #
#         # # branch4_xyz, branch4_idx = random_sample(branch3_xyz, 128)
#         #
#         # branch4_points_FP_global = self.globalTrans_stage2_la4(branch4_points_FP)
#         # branch4_points_FP_local = self.localTrans_stage2_la4(branch4_points_FP, branch3_xyz, branch4_xyz, branch3_points_FP, branch4_idx_FP)
#         # branch4_points_FP_cat = torch.cat((branch4_points_FP_local, branch4_points_FP_global), dim=1)
#         # branch4_points_FP = self.stage24(branch4_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch4_points_FP
#         #
#         # # branch4_points = index_points(branch3_points_FP.permute(0,2,1).contiguous(), branch4_idx_FP).permute(0,2,1)
#         # # branch4_points = self.globalTrans_stage2_la4(branch4_points)
#         # # branch4_points_FP = branch4_points_FP + branch4_points
#         # # branch1_points = self.localTrans_stage2_la1(branch1_points, branch1_points)
#         # # branch2_points_FP = self.localTrans_stage2_la2(branch2_points_FP, branch2_points_FP)
#         # # branch3_points_FP = self.localTrans_stage2_la3(branch3_points_FP, branch3_points_FP)
#         # # branch4_points_FP = self.localTrans_stage2_la4(branch4_points_FP, branch4_points_FP)
#         #
#         # ##################  Fuse
#         #
#         # # branch1_xyz_f, idx1 = self.localTrans_stage2_xyz_la1(branch1_xyz, branch1_xyz)
#         # # branch2_xyz_f, idx2 = self.localTrans_stage2_xyz_la2(branch2_xyz, branch2_xyz)
#         # # branch3_xyz_f, idx3 = self.localTrans_stage2_xyz_la3(branch3_xyz, branch3_xyz)
#         # # branch4_xyz_f, idx4 = self.localTrans_stage2_xyz_la4(branch4_xyz, branch4_xyz)
#         #
#         # # branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage2_fuse_atten(branch1_xyz_f, branch1_points, branch2_xyz_f, branch2_points_FP,
#         # #                                                                                                  branch3_xyz_f, branch3_points_FP, branch4_xyz_f, branch4_points_FP)
#         #
#         # branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage2_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
#         #                                                                                      branch2_idx_FP, branch3_idx_FP, branch4_idx_FP, branch1_xyz, branch2_xyz, branch3_xyz, branch4_xyz)
#         #
#         # # x2 = self.conv_x2(branch1_points.unsqueeze(-1).contiguous())
#         # x2 = branch1_points.unsqueeze(-1).contiguous()
#         # # x2 = self.conv_x2(branch4_points_FP.unsqueeze(-1).contiguous())
#         #
#         #
#         #
#         #
#         # ############## Local Aggregation  stage3
#         # # branch1_points = self.stage3_la1(branch1_points, branch1_xyz, norm=branch1_norm)
#         # # branch2_points_FP = self.stage3_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
#         # # branch3_points_FP = self.stage3_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
#         # # branch4_points_FP = self.stage3_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)
#         #
#         #
#         # # branch1_points = self.stage3_localGraph1(branch1_points)
#         # # branch2_points_FP = self.stage3_localGraph2(branch2_points_FP)
#         # # branch3_points_FP = self.stage3_localGraph3(branch3_points_FP)
#         # # branch4_points_FP = self.stage3_localGraph4(branch4_points_FP)
#         #
#         # branch1_points_global = self.globalTrans_stage3_la1(branch1_points)
#         # branch1_points_local = self.localTrans_stage3_la1(branch1_points, branch1_xyz, branch1_xyz, branch1_points)
#         # branch1_points_cat = torch.cat((branch1_points_local, branch1_points_global), dim=1)
#         # branch1_points = self.stage31(branch1_points_cat.unsqueeze(-1)).squeeze(-1) + branch1_points
#         #
#         #
#         # branch2_points_FP_global = self.globalTrans_stage3_la2(branch2_points_FP)
#         # branch2_points_FP_local = self.localTrans_stage3_la2(branch2_points_FP, branch1_xyz, branch2_xyz, branch1_points, branch2_idx_FP)
#         # branch2_points_FP_cat = torch.cat((branch2_points_FP_local, branch2_points_FP_global), dim=1)
#         # branch2_points_FP = self.stage32(branch2_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch2_points_FP
#         #
#         #
#         # branch3_points_FP_global = self.globalTrans_stage3_la3(branch3_points_FP)
#         # branch3_points_FP_local = self.localTrans_stage3_la3(branch3_points_FP, branch2_xyz, branch3_xyz, branch2_points_FP, branch3_idx_FP)
#         # branch3_points_FP_cat = torch.cat((branch3_points_FP_local, branch3_points_FP_global), dim=1)
#         # branch3_points_FP = self.stage33(branch3_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch3_points_FP
#         #
#         #
#         #
#         # branch4_points_FP_global = self.globalTrans_stage3_la4(branch4_points_FP)
#         # branch4_points_FP_local = self.localTrans_stage3_la4(branch4_points_FP, branch3_xyz, branch4_xyz, branch3_points_FP, branch4_idx_FP)
#         # branch4_points_FP_cat = torch.cat((branch4_points_FP_local, branch4_points_FP_global), dim=1)
#         # branch4_points_FP = self.stage34(branch4_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch4_points_FP
#         #
#         #
#         #
#         # # branch1_points = self.globalTrans_stage3_la1(branch1_points, branch1_xyz, branch1_xyz, branch1_points)
#         # #
#         # # branch2_points = index_points(branch1_points.permute(0, 2, 1).contiguous(), branch2_idx_FP).permute(0,2,1)
#         # # branch2_points = self.globalTrans_stage3_la2(branch2_points)
#         # # branch2_points_FP = branch2_points_FP + branch2_points
#         # #
#         # # branch3_points = index_points(branch2_points_FP.permute(0, 2, 1).contiguous(), branch3_idx_FP).permute(0,2,1)
#         # # branch3_points = self.globalTrans_stage3_la3(branch3_points)
#         # # branch3_points_FP = branch3_points_FP + branch3_points
#         # #
#         # # branch4_points = index_points(branch3_points_FP.permute(0, 2, 1).contiguous(), branch4_idx_FP).permute(0,2,1)
#         # # branch4_points = self.globalTrans_stage3_la4(branch4_points)
#         # # branch4_points_FP = branch4_points_FP + branch4_points
#         #
#         #
#         # # branch1_xyz_f, _ = self.localTrans_stage3_xyz_la1(branch1_xyz, branch1_xyz, idx1)
#         # # branch2_xyz_f, _ = self.localTrans_stage3_xyz_la2(branch2_xyz, branch2_xyz, idx2)
#         # # branch3_xyz_f, _ = self.localTrans_stage3_xyz_la3(branch3_xyz, branch3_xyz, idx3)
#         # # branch4_xyz_f, _ = self.localTrans_stage3_xyz_la4(branch4_xyz, branch4_xyz, idx4)
#         #
#         #
#         # # branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage3_fuse_atten(branch1_xyz_f, branch1_points, branch2_xyz_f, branch2_points_FP,
#         # #                                                                                                  branch3_xyz_f, branch3_points_FP, branch4_xyz_f, branch4_points_FP)
#         #
#         # # branch1_points = self.localTrans_stage3_la1(branch1_points, branch1_points)
#         # # branch2_points_FP = self.localTrans_stage3_la2(branch2_points_FP, branch2_points_FP)
#         # # branch3_points_FP = self.localTrans_stage3_la3(branch3_points_FP, branch3_points_FP)
#         # # branch4_points_FP = self.localTrans_stage3_la4(branch4_points_FP, branch4_points_FP)
#         #
#         #
#         # ##################  Fuse
#         # branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage3_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
#         #                                                                                      branch2_idx_FP, branch3_idx_FP, branch4_idx_FP, branch1_xyz, branch2_xyz, branch3_xyz, branch4_xyz)
#         #
#         # # x3 = self.conv_x3(branch1_points.unsqueeze(-1).contiguous())
#         # x3 = branch1_points.unsqueeze(-1).contiguous()
#         # # x3 = self.conv_x3(branch4_points_FP.unsqueeze(-1).contiguous())
#         #
#         #
#         #
#         #
#         #
#         # ############## Local Aggregation  stage4
#         # # branch1_points = self.stage4_la1(branch1_points, branch1_xyz, norm=branch1_norm)
#         # # branch2_points_FP = self.stage4_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
#         # # branch3_points_FP = self.stage4_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
#         # # branch4_points_FP = self.stage4_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)
#         #
#         #
#         # # branch1_points = self.stage4_localGraph1(branch1_points)
#         # # branch2_points_FP = self.stage4_localGraph2(branch2_points_FP)
#         # # branch3_points_FP = self.stage4_localGraph3(branch3_points_FP)
#         # # branch4_points_FP = self.stage4_localGraph4(branch4_points_FP)
#         #
#         # branch1_points_global = self.globalTrans_stage4_la1(branch1_points)
#         # branch1_points_local = self.localTrans_stage4_la1(branch1_points, branch1_xyz, branch1_xyz, branch1_points)
#         # branch1_points_cat = torch.cat((branch1_points_local, branch1_points_global), dim=1)
#         # branch1_points = self.stage41(branch1_points_cat.unsqueeze(-1)).squeeze(-1) + branch1_points
#         #
#         # branch2_points_FP_global = self.globalTrans_stage4_la2(branch2_points_FP)
#         # branch2_points_FP_local = self.localTrans_stage4_la2(branch2_points_FP, branch1_xyz, branch2_xyz, branch1_points, branch2_idx_FP)
#         # branch2_points_FP_cat = torch.cat((branch2_points_FP_local, branch2_points_FP_global), dim=1)
#         # branch2_points_FP = self.stage42(branch2_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch2_points_FP
#         #
#         # branch3_points_FP_global = self.globalTrans_stage4_la3(branch3_points_FP)
#         # branch3_points_FP_local = self.localTrans_stage4_la3(branch3_points_FP, branch2_xyz, branch3_xyz, branch2_points_FP, branch3_idx_FP)
#         # branch3_points_FP_cat = torch.cat((branch3_points_FP_local, branch3_points_FP_global), dim=1)
#         # branch3_points_FP = self.stage43(branch3_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch3_points_FP
#         #
#         # branch4_points_FP_global = self.globalTrans_stage4_la4(branch4_points_FP)
#         # branch4_points_FP_local = self.localTrans_stage4_la4(branch4_points_FP, branch3_xyz, branch4_xyz, branch3_points_FP, branch4_idx_FP)
#         # branch4_points_FP_cat = torch.cat((branch4_points_FP_local, branch4_points_FP_global), dim=1)
#         # branch4_points_FP = self.stage44(branch4_points_FP_cat.unsqueeze(-1)).squeeze(-1) + branch4_points_FP
#         #
#         #
#         # # branch1_points = self.globalTrans_stage4_la1(branch1_points, branch1_xyz, branch1_xyz, branch1_points)
#         # #
#         # # branch2_points = index_points(branch1_points.permute(0, 2, 1).contiguous(), branch2_idx_FP).permute(0,2,1)
#         # # branch2_points = self.globalTrans_stage4_la2(branch2_points, branch1_xyz, branch2_xyz, branch1_points)
#         # # branch2_points_FP = branch2_points_FP + branch2_points
#         # #
#         # # branch3_points = index_points(branch2_points_FP.permute(0, 2, 1).contiguous(), branch3_idx_FP).permute(0,2,1)
#         # # branch3_points = self.globalTrans_stage4_la3(branch3_points, branch2_xyz, branch3_xyz, branch2_points)
#         # # branch3_points_FP = branch3_points_FP + branch3_points
#         # #
#         # # branch4_points = index_points(branch3_points_FP.permute(0, 2, 1).contiguous(), branch4_idx_FP).permute(0,2,1)
#         # # branch4_points = self.globalTrans_stage4_la4(branch4_points, branch3_xyz, branch4_xyz, branch3_points)
#         # # branch4_points_FP = branch4_points_FP + branch4_points
#         #
#         #
#         # # branch1_xyz_f, _ = self.localTrans_stage4_xyz_la1(branch1_xyz, branch1_xyz, idx1)
#         # # branch2_xyz_f, _ = self.localTrans_stage4_xyz_la2(branch2_xyz, branch2_xyz, idx2)
#         # # branch3_xyz_f, _ = self.localTrans_stage4_xyz_la3(branch3_xyz, branch3_xyz, idx3)
#         # # branch4_xyz_f, _ = self.localTrans_stage4_xyz_la4(branch4_xyz, branch4_xyz, idx4)
#         #
#         # # branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage4_fuse_atten(branch1_xyz_f, branch1_points, branch2_xyz_f, branch2_points_FP,
#         # #                                                                                                  branch3_xyz_f, branch3_points_FP, branch4_xyz_f, branch4_points_FP)
#         #
#         # # branch1_points = self.localTrans_stage4_la1(branch1_points, branch1_points)
#         # # branch2_points_FP = self.localTrans_stage4_la2(branch2_points_FP, branch2_points_FP)
#         # # branch3_points_FP = self.localTrans_stage4_la3(branch3_points_FP, branch3_points_FP)
#         # # branch4_points_FP = self.localTrans_stage4_la4(branch4_points_FP, branch4_points_FP)
#         #
#         #
#         # ##################  Fuse
#         # branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage4_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
#         #                                                                                      branch2_idx_FP, branch3_idx_FP, branch4_idx_FP, branch1_xyz, branch2_xyz, branch3_xyz, branch4_xyz)
#         #
#         # # x4 = self.conv_x4(branch1_points.unsqueeze(-1).contiguous())
#         # x4 = branch1_points.unsqueeze(-1).contiguous()
#         # # x4 = self.conv_x4(branch4_points_FP.unsqueeze(-1).contiguous())
#
#         # branch1_points = self.head1(branch1_points.unsqueeze(-1)).squeeze(-1)
#         # branch2_points_FP = self.head2(branch2_points_FP.unsqueeze(-1)).squeeze(-1)
#         # branch3_points_FP = self.head3(branch3_points_FP.unsqueeze(-1)).squeeze(-1)
#         # branch4_points_FP = self.head4(branch4_points_FP.unsqueeze(-1)).squeeze(-1)
#         #
#         # branch1_points = index_points(branch1_points.permute(0,2,1), branch2_idx_FP).permute(0, 2, 1)
#         # branch2_points_FP = branch2_points_FP + self.head12(branch1_points.unsqueeze(-1)).squeeze(-1)
#         #
#         # branch2_points_FP = index_points(branch2_points_FP.permute(0,2,1), branch3_idx_FP).permute(0, 2, 1)
#         # branch3_points_FP = branch3_points_FP + self.head22(branch2_points_FP.unsqueeze(-1)).squeeze(-1)
#         #
#         # branch3_points_FP = index_points(branch3_points_FP.permute(0,2,1), branch4_idx_FP).permute(0, 2, 1)
#         # branch4_points_FP = branch4_points_FP + self.head32(branch3_points_FP.unsqueeze(-1)).squeeze(-1)
#         #
#         #
#         # final = branch4_points_FP
#         #
#         # x1 = F.adaptive_max_pool1d(final, 1)
#         # x2 = F.adaptive_avg_pool1d(final, 1)
#         # final_fuse = torch.cat((x1, x2), 1).unsqueeze(-1)
#         #
#         # final_fuse = self.final_class(final_fuse).squeeze(-1).squeeze(-1)
#
#         final = self.final(torch.cat((x1, x2, x3, x4), 2)).permute(0, 2, 1).contiguous()
#
#         x1 = F.adaptive_max_pool1d(final, 1)
#         x2 = F.adaptive_avg_pool1d(final, 1)
#         final_fuse = torch.cat((x1, x2), 1).squeeze(-1)
#
#         final_fuse = self.lrelu(self.bn(self.final_class(final_fuse)))
#         # final_fuse = self.drop(final_fuse)
#
#         return xyz, final_fuse, ret2, ret3, ret4
#         # return xyz, final_fuse, d1, d2, d3, d4, d5, d1_idx, d2_idx, d3_idx, d4_idx, d5_idx

def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def resort_points(points, idx):
    """
    Resort Set of points along G dim

    """
    device = points.device
    B, N, G, _ = points.shape

    view_shape = [B, 1, 1]
    repeat_shape = [1, N, G]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points

def group_by_umbrella(xyz, new_xyz, k=9, cuda=False):
    """
    Group a set of points into umbrella surfaces

    """
    idx = query_knn_point(k, xyz, new_xyz, cuda=cuda)
    torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
    torch.cuda.empty_cache()

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N', K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1]

    # [B, N', K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz

class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella-based Surface Abstraction Module

    """

    def __init__(self, k, in_channel, aggr_type='sum', return_dist=False, random_inv=True, cuda=False):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type = aggr_type
        self.cuda = cuda

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        )

    def forward(self, center):
        center = center.permute(0, 2, 1)
        # surface construction
        group_xyz = group_by_umbrella(center, center, k=self.k, cuda=self.cuda)  # [B, N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center)
        if self.return_dist:
            group_pos = cal_const(group_normal, group_center)
            group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
            new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        else:
            group_normal, group_center = check_nan_umb(group_normal, group_center)
            new_feature = torch.cat([group_center, group_polar, group_normal], dim=-1)
        new_feature = new_feature.permute(0, 3, 2, 1)  # [B, C, G, N]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        if self.aggr_type == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature

#######################################################################################################

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, act=True):
        super(Linear, self).__init__()

        self.act_flag = act
        self.bn_flag = bn

        self.linear = nn.Linear(in_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):

        out = self.linear(input)

        if self.bn_flag is True:
            out = self.norm1(out)
        else:
            out = self.norm2(out.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        if self.act_flag is True:
            out = self.act(out)

        return out

class LocalMerge(nn.Module):
    def __init__(self, in_channels, out_channels, knn, usetanh=False, residual=False, cuda=False):
        super(LocalMerge, self).__init__()

        self.knn = knn
        self.cuda = cuda
        self.usetanh = usetanh
        self.residual = residual
        # self.fc1 = Linear(out_channels*2, out_channels, bn=False)
        # self.fc2 = Linear(out_channels*3, out_channels, bn=False)


        self.xyz_Trans = LocalTrans(3, out_channels, knn, usetanh=self.usetanh, residual=True, cuda=self.cuda)
        self.normal_Trans = LocalTrans(10, out_channels, knn, usetanh=self.usetanh, residual=True, cuda=self.cuda)
        self.feature_Trans = LocalTrans(in_channels, out_channels, knn, usetanh=self.usetanh, residual=self.residual, cuda=self.cuda)

    def forward(self, xyz, base_xyz, normal, feature=None, FPS_idx=None):


        idx = query_knn_point(self.knn, base_xyz, xyz, cuda=self.cuda)  ### B N K
        torch.cuda.empty_cache()

        if feature is None:
            xyz_f = self.xyz_Trans(features=xyz, idx=idx, pos=base_xyz, FPS_idx=FPS_idx, xyz=True)
            # normal_f = self.normal_Trans(features=normal, idx=idx, FPS_idx=FPS_idx)

            merge_features = xyz_f
            # merge_features = self.fc1(torch.cat((xyz_f, normal_f), dim=2))

        else:

            # xyz_f = self.xyz_Trans(features=base_xyz, idx=idx, FPS_idx=FPS_idx, xyz=True)
            # normal_f = self.normal_Trans(features=normal, idx=idx, FPS_idx=FPS_idx)
            merge_features = self.feature_Trans(features=feature, idx=idx, pos=base_xyz, FPS_idx=FPS_idx)

            # merge_features = self.fc2(torch.cat((xyz_f, normal_f, features), dim=2))

            if FPS_idx is not None:
                normal = index_points(normal, FPS_idx, cuda=self.cuda, is_group=False)
                torch.cuda.empty_cache()

        return merge_features, normal

class LocalTrans(nn.Module):
    def __init__(self, in_c, out_c, patch_num, usetanh=False, residual=False, cuda=False):
        super(LocalTrans, self).__init__()

        self.patchNum = patch_num
        self.residual = residual
        self.usetanh = usetanh
        self.cuda = cuda

        # self.fc_pre = nn.Linear(in_c, out_c)
        # self.fc2 = nn.Linear(out_c, out_c)

        # self.pos_emb1 = nn.Sequential(
        #     nn.Linear(3, out_c),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(out_c, out_c)
        # )
        #
        # self.pos_emb2 = nn.Sequential(
        #     nn.Linear(3, out_c),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(out_c, out_c)
        # )


        # self.pos_emb1 = Linear(3, out_c, bn=False)
        # self.pos_emb2 = Linear(3, out_c, bn=False)

        self.q = nn.Linear(in_c, out_c)
        self.k = nn.Linear(in_c, out_c)
        self.v = nn.Linear(in_c, out_c)

        # self.trans = nn.Linear(out_c, out_c)

        self.conv_res = Linear(in_c, out_c, bn=False)

        # self.drop = nn.Dropout(0.1)     ########################
        self.ffn = Linear(out_c, out_c, bn=False)

        # self.pos_emb = Linear(3, in_c, bn=bn)

        # self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def forward(self, features, idx, pos, FPS_idx=None, xyz=False):

        # pos_emb = self.pos_emb(base_xyz)
        # features = features + pos_emb


        if FPS_idx is not None:
            residual = index_points(features, FPS_idx, cuda=self.cuda, is_group=False)
            torch.cuda.empty_cache()
            pos_emb = index_points(pos, FPS_idx, cuda=self.cuda, is_group=False)
            torch.cuda.empty_cache()

            # features = self.fc_pre(features)
            # center_features = index_points(features, FPS_idx, cuda=self.cuda, is_group=False)
            center_features = residual

        else:
            residual = features
            pos_emb = pos

            # features = self.fc_pre(features)
            center_features = features



        if self.residual is True:
            residual = self.conv_res(residual)

        knn_xyz = index_points(pos, idx, cuda=self.cuda, is_group=True)
        torch.cuda.empty_cache()

        if xyz is True:
            local_query = self.q(center_features).unsqueeze(-2)


            local_key = index_points(features, idx[:, :, 0:self.patchNum:1], cuda=self.cuda, is_group=True)#.permute(0,1,3,2).contiguous()
            torch.cuda.empty_cache()
            local_key = local_key - center_features.unsqueeze(-2)
            local_key = self.k(local_key)#.permute(0,1,3,2).contiguous()


            local_value = index_points(features, idx[:, :, 0:self.patchNum:1], cuda=self.cuda, is_group=True)
            torch.cuda.empty_cache()
            local_value = local_value - center_features.unsqueeze(-2)
            local_value = self.v(local_value)
            # anchor_value = self.v(center_features)

            # energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
            # energy = energy / np.sqrt(energy.size(-1))

            # B, N, K, C = local_key.shape
            # pos_emb = self.fc_delta(pos_emb.unsqueeze(-2) - knn_xyz)
            # energy = self.fc_gamma(local_query - local_key + pos_emb)
            # energy = (local_query - local_key) * self.pos_emb1(pos_emb.unsqueeze(-2) - knn_xyz) + self.pos_emb2(pos_emb.unsqueeze(-2) - knn_xyz)
            energy = local_query - local_key #+ pos_emb
            # energy = self.trans(torch.cat((local_query.expand(-1,-1,self.patchNum,-1), local_key), dim=-1))

            if self.usetanh is True:
                attention = self.tanh(energy) / self.patchNum
                context = torch.matmul(attention, local_value).squeeze(-2) #+ anchor_value  ###nn.bmm
            else:
                attention = F.softmax(energy/np.sqrt(local_key.size(-1)), dim=-2)
                # attention = self.tanh(attention)
                # attention = F.sigmoid(energy / np.sqrt(local_key.size(-1)))
                offset = torch.sum(attention, dim=2, keepdim=True)
                attention = attention - offset
                # context = torch.matmul(attention, local_value).squeeze(-2)  ###nn.bmm
                # context = torch.einsum('bnkc,bnkc->bnc', attention, local_value + pos_emb)
                context = attention * (local_value)# + pos_emb)
                context = torch.max(context, 2)[0]
                # context = F.normalize(context, p=2.0, dim=-1)
                # merge_features = merge_features.permute(0, 2, 1)


        else:

            local_query = self.q(center_features).unsqueeze(-2)

            local_key = self.k(features)
            local_key = index_points(local_key, idx[:, :, 0:self.patchNum:1], cuda=self.cuda, is_group=True)#.permute(0,1,3,2).contiguous()
            torch.cuda.empty_cache()

            local_value = self.v(features)
            local_value = index_points(local_value, idx[:, :, 0:self.patchNum:1], cuda=self.cuda, is_group=True)
            torch.cuda.empty_cache()
            # anchor_value = self.v(center_features)

            # energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
            # energy = energy / np.sqrt(energy.size(-1))
            # pos_emb = self.fc_delta(pos_emb.unsqueeze(-2) - knn_xyz)
            # energy = self.fc_gamma(local_query - local_key + pos_emb)
            # energy = (local_query - local_key) * self.pos_emb1(pos_emb.unsqueeze(-2) - knn_xyz) + self.pos_emb2(pos_emb.unsqueeze(-2) - knn_xyz)
            energy = local_query - local_key #+ pos_emb
            # energy = self.trans(torch.cat((local_query.expand(-1,-1,self.patchNum,-1), local_key), dim=-1))



            if self.usetanh is True:
                attention = self.tanh(energy) / self.patchNum
                context = torch.matmul(attention, local_value).squeeze(-2) #+ anchor_value  ###nn.bmm
            else:
                attention = F.softmax(energy/np.sqrt(local_key.size(-1)), dim=-2)
                # attention = self.tanh(attention)
                # attention = F.sigmoid(energy / np.sqrt(local_key.size(-1)))
                offset = torch.sum(attention, dim=2, keepdim=True)
                attention = attention - offset
                # context = torch.matmul(attention, local_value).squeeze(-2)  ###nn.bmm
                # context = torch.einsum('bnkc,bnkc->bnc', attention, local_value + pos_emb)
                context = attention * (local_value)# + pos_emb)
                context = torch.max(context, 2)[0]
                # context = F.normalize(context, p=2.0, dim=-1)

        # context = residual + self.ffn(residual - context)
        context = residual + self.ffn(context)

        # if FPS_idx is not None:
        #     residual = index_points(features, FPS_idx, cuda=self.cuda, is_group=False)
        #     torch.cuda.empty_cache()
        #
        #     center_features = residual
        #
        #     # if self.residual is True:
        #     #     residual = self.conv_res(residual.unsqueeze(-1)).squeeze(-1)
        #
        #     if knn is not None:
        #         idx = knn
        #     else:
        #         # dist, idx = knn_point(self.patchNum, base_xyz, xyz)  ### B N K
        #         idx = query_knn_point(self.patchNum, base_xyz, xyz, cuda=self.cuda)  ### B N K
        #         torch.cuda.empty_cache()
        #
        #     ########################################################
        #     local_query = self.q3(center_features).unsqueeze(-2)
        #
        #     local_key = self.k3(features)
        #     local_key = index_points(local_key, idx[:, :, 0:self.patchNum:1], cuda=self.cuda, is_group=True).permute(0, 1, 3, 2).contiguous()
        #     torch.cuda.empty_cache()
        #
        #     local_value = self.v3(features)
        #     local_value = index_points(local_value, idx[:, :, 0:self.patchNum:1], cuda=self.cuda, is_group=True)
        #     torch.cuda.empty_cache()
        #     anchor_value = self.v3(center_features)
        #
        #     energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
        #     energy = energy / np.sqrt(energy.size(-1))
        #
        #     if self.usetanh is True:
        #         attention = self.tanh(energy) / self.patchNum
        #     else:
        #         attention = self.softmax(energy)
        #
        #     if self.anchorV is True:
        #         context = torch.matmul(attention, local_value).squeeze(-2) + anchor_value  ###nn.bmm
        #     else:
        #         context = torch.matmul(attention, local_value).squeeze(-2)  ###nn.bmm
        #
        #
        #     context = residual + self.ffn(residual - context)
        #
        # else:
        #     residual = features
        #     center_features = residual
        #
        #     if self.residual is True:
        #         residual = self.conv_res(residual)
        #
        #     if knn is not None:
        #         idx = knn
        #     else:
        #         # dist, idx = knn_point(self.patchNum, base_xyz, xyz)  ### B N K
        #         idx = query_knn_point(self.patchNum, base_xyz, xyz, cuda=self.cuda)  ### B N K
        #         torch.cuda.empty_cache()
        #
        #     ########################################################
        #     local_query = self.q3(center_features).unsqueeze(-2)
        #
        #     local_key = self.k3(features)
        #     local_key = index_points(local_key, idx[:, :, 0:self.patchNum:1], cuda=self.cuda, is_group=True).permute(0, 1, 3, 2).contiguous()
        #     torch.cuda.empty_cache()
        #
        #     local_value = self.v3(features)
        #     local_value = index_points(local_value, idx[:, :, 0:self.patchNum:1], cuda=self.cuda, is_group=True)
        #     torch.cuda.empty_cache()
        #     anchor_value = self.v3(center_features)
        #
        #     energy = torch.matmul(local_query, local_key)  # + pos_encq ###nn.bmm
        #     energy = energy / np.sqrt(energy.size(-1))
        #
        #     if self.usetanh is True:
        #         attention = self.tanh(energy) / self.patchNum
        #     else:
        #         attention = self.softmax(energy)
        #
        #     if self.anchorV is True:
        #         context = torch.matmul(attention, local_value).squeeze(-2) + anchor_value
        #     else:
        #         context = torch.matmul(attention, local_value).squeeze(-2)  ###nn.bmm
        #
        #
        #     context = residual + self.ffn(residual - context)


        return context

class KeepHighResolutionModulePartSeg(nn.Module):

    def __init__(self, data_C, b1_C, b2_C, b3_C, b4_C, cuda=False):
        super(KeepHighResolutionModulePartSeg, self).__init__()

        # self.local_num_neighbors = [16, 32]
        self.neighbour = 16
        self.cuda = cuda


        # self.Trans_pool2 = LocalTrans(64, 128, 16, usetanh=False, residual=True, anchorV=False, bn=False)
        # self.pool2 = IndexSelect(128, 16)
        #
        # self.Trans_pool3 = LocalTrans(128, 256, 16, usetanh=False, residual=True, anchorV=False, bn=False)
        # self.pool3 = IndexSelect(256, 16)
        #
        # self.Trans_pool4 = LocalTrans(256, 512, 16, usetanh=False, residual=True, anchorV=False, bn=False)
        # self.pool4 = IndexSelect(512, 16)


        self.la0 = LocalMerge(64, 64, 8, usetanh=False, residual=False, cuda=self.cuda)
        self.la1 = LocalMerge(64, 64, 8, usetanh=False, residual=False, cuda=self.cuda)
        self.la2 = LocalMerge(64, 64, 8, usetanh=False, residual=False, cuda=self.cuda)

        self.la3 = LocalMerge(64, 128, 8, usetanh=False, residual=True, cuda=self.cuda)
        self.la4 = LocalMerge(128, 256, 8, usetanh=False, residual=True, cuda=self.cuda)
        self.la5 = LocalMerge(256, 512, 8, usetanh=False, residual=True, cuda=self.cuda)

        self.upla4 = LocalMerge(512, 512, 8, usetanh=False, residual=False, cuda=self.cuda)
        self.upla3 = LocalMerge(256, 256, 8, usetanh=False, residual=False, cuda=self.cuda)
        self.upla2 = LocalMerge(128, 128, 8, usetanh=False, residual=False, cuda=self.cuda)
        self.upla1 = LocalMerge(64, 64, 8, usetanh=False, residual=False, cuda=self.cuda)
        self.upla0 = LocalMerge(64, 64, 8, usetanh=False, residual=False, cuda=self.cuda)

        self.up5_4 = PointNetFeaturePropagation(512, [256, ], act=True, cuda=self.cuda)
        self.up4_3 = PointNetFeaturePropagation(256, [128, ], act=True, cuda=self.cuda)
        self.up3_2 = PointNetFeaturePropagation(128, [64, ], act=True, cuda=self.cuda)
        self.up2_1 = PointNetFeaturePropagation(64, [64, ], act=True, cuda=self.cuda)
        self.up1_0 = PointNetFeaturePropagation(64, [64, ], act=True, cuda=self.cuda)

        # ###stage2
        # self.Trans_stage2_la11 = LocalTrans(64, 64, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage2_la12 = LocalTrans(64, 64, 16, usetanh=True)
        #
        # self.Trans_stage2_la21 = LocalTrans(128, 128, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage2_la22 = LocalTrans(128, 128, 16, usetanh=True)
        #
        # self.stage2_fuse = Fuse_Stage(64, 128, 256, 512)
        #
        # ###stage3
        # # self.stage3_trans = SharedMLP(128, 256, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        #
        # self.Trans_stage3_la11 = LocalTrans(64, 64, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage3_la12 = LocalTrans(64, 64, 16, usetanh=True)
        #
        # self.Trans_stage3_la21 = LocalTrans(128, 128, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage3_la22 = LocalTrans(128, 128, 16, usetanh=True)
        #
        # self.Trans_stage3_la31 = LocalTrans(256, 256, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage3_la32 = LocalTrans(256, 256, 16, usetanh=True)
        #
        # self.stage3_fuse = Fuse_Stage(64, 128, 256, 512)
        #
        # ###stage4
        # # self.stage4_trans = SharedMLP(256, 512, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        #
        # self.Trans_stage4_la11 = LocalTrans(64, 64, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage4_la12 = LocalTrans(64, 64, 16, usetanh=True)
        #
        # self.Trans_stage4_la21 = LocalTrans(128, 128, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage4_la22 = LocalTrans(128, 128, 16, usetanh=True)
        #
        # self.Trans_stage4_la31 = LocalTrans(256, 256, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage4_la32 = LocalTrans(256, 256, 16, usetanh=True)
        #
        # self.Trans_stage4_la41 = LocalTrans(512, 512, 16, usetanh=True, anchorV=True)
        # # self.Trans_stage4_la42 = LocalTrans(512, 512, 16, usetanh=True)
        #
        # self.stage4_fuse = Fuse_Stage(64, 128, 256, 512)



        # self.conv_x1 = Linear(64, 128, bn=False)
        # # self.fuse1_x1 = PointNetFeaturePropagation(b1_C, [b1_C, ], act=True)
        # # self.fuse2_x1 = PointNetFeaturePropagation(b1_C, [b1_C, ], act=True)
        # self.fuse1_x1 = Upsample(b1_C, b1_C, 8)   ####
        # self.fuse2_x1 = Upsample(b1_C, b1_C, 8)  ####
        # #
        # self.conv_x2 = Linear(64, 128, bn=False)
        # # self.fuse1_x2 = PointNetFeaturePropagation(b1_C, [b1_C, ], act=True)
        # # self.fuse2_x2 = PointNetFeaturePropagation(b1_C, [b1_C, ], act=True)
        # self.fuse1_x2 = Upsample(b1_C, b1_C, 8)   ####
        # self.fuse2_x2 = Upsample(b1_C, b1_C, 8)   ####
        # #
        # self.conv_x3 = Linear(64, 128, bn=False)
        # # self.fuse1_x3 = PointNetFeaturePropagation(b1_C, [b1_C, ], act=True)
        # # self.fuse2_x3 = PointNetFeaturePropagation(b1_C, [b1_C, ], act=True)
        # self.fuse1_x3 = Upsample(b1_C, b1_C, 8)    ####
        # self.fuse2_x3 = Upsample(b1_C, b1_C, 8)    ####
        #
        # self.conv_x4 = Linear(64, 128, bn=False)
        # # self.fuse1_x4 = PointNetFeaturePropagation(b1_C, [b1_C, ], act=True)
        # # self.fuse2_x4 = PointNetFeaturePropagation(b1_C, [b1_C, ], act=True)
        # self.fuse1_x4 = Upsample(b1_C, b1_C, 8)    ####
        # self.fuse2_x4 = Upsample(b1_C, b1_C, 8)    ####



        # self.fuse43 = PointNetFeaturePropagation(b4_C, [b3_C, ])
        # self.conv43 = Linear(b3_C*2, b3_C, bn=False)
        #
        # self.fuse32 = PointNetFeaturePropagation(b3_C, [b2_C, ])
        # self.conv32 = Linear(b2_C*2, b2_C, bn=False)
        #
        # self.fuse21 = PointNetFeaturePropagation(b2_C, [b1_C, ])
        # self.conv21 = Linear(b1_C*2, b1_C, bn=False)
        #
        # self.fuse10 = PointNetFeaturePropagation(b1_C, [b1_C, ])
        # self.conv10 = Linear(b1_C*2, b1_C, bn=False)
        #
        # self.fuse0 = PointNetFeaturePropagation(b1_C, [b1_C, ])
        # self.conv0 = Linear(b1_C*2, b1_C*2, bn=False)


        self.conv6 = Linear(64, 256, bn=False)

        self.conv7 = Linear(16, 64, bn=False)

        # self.final_class = nn.Linear(1024, 1024)
        # self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, xyz, normal, label):

        xyz = xyz.permute(0, 2, 1).contiguous()
        normal = normal.permute(0, 2, 1).contiguous()

        _, num_points, _ = xyz.shape

        points_FPS, normal = self.la0(xyz=xyz, base_xyz=xyz, normal=normal)

        # value0, idx0 = self.pool0(xyz.permute(0,2,1), points)
        # FPS_xyz0 = index_points(xyz.permute(0,2,1), idx0)

        FPS_idx0 = farthest_point_sample(points_FPS, 1024, cuda=self.cuda)
        torch.cuda.empty_cache()
        FPS_xyz0 = index_points(xyz, FPS_idx0, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        # FPS_xyz0, _ = self.FPS_offset1(xyz.permute(0, 2, 1), points, FPS_idx0, FPS_xyz0_gt)
        # FPS_xyz0, FPS_idx0 = self.FPS_offset1(xyz.permute(0,2,1), points)
        # d1 = self.diffusion1(FPS_xyz0, xyz.permute(0,2,1), FPS_idx0)
        # d1_idx = FPS_idx0

        # points0 = self.Trans0(xyz=FPS_xyz0, base_xyz=xyz.permute(0, 2, 1).contiguous(), features=points, FPS_idx=FPS_idx0)
        points0_FPS, normal = self.la1(xyz=FPS_xyz0, base_xyz=xyz, normal=normal, feature=points_FPS, FPS_idx=FPS_idx0)

        # umber_feature0 = index_points(features.permute(0, 2, 1).contiguous(), FPS_idx0, cuda=self.cuda, is_group=False)
        # torch.cuda.empty_cache()
        # points0 = self.lrelu(points0 + self.start2(umber_feature0))

        FPS_idx1 = farthest_point_sample(points0_FPS, 512, cuda=self.cuda)
        torch.cuda.empty_cache()
        FPS_xyz1 = index_points(FPS_xyz0, FPS_idx1, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        # FPS_xyz1, _ = self.FPS_offset2(FPS_xyz0_gt, points0, FPS_idx1, FPS_xyz1_gt)
        # FPS_xyz1, FPS_idx1 = self.FPS_offset2(FPS_xyz0, points0)
        # d2 = self.diffusion2(FPS_xyz1, FPS_xyz0, FPS_idx1)
        # d2_idx = FPS_idx1

        # point1, normal = self.Trans1(xyz=FPS_xyz1, base_xyz=FPS_xyz0, features=points0, FPS_idx=FPS_idx1)
        point1, normal = self.la2(xyz=FPS_xyz1, base_xyz=FPS_xyz0, normal=normal, feature=points0_FPS, FPS_idx=FPS_idx1)
        # umber_feature1 = index_points(umber_feature0, FPS_idx1, cuda=self.cuda, is_group=False)
        # torch.cuda.empty_cache()
        # point1 = self.lrelu(point1 + self.start3(umber_feature1))

        branch1_xyz = FPS_xyz1
        branch1_points_FP = point1

        idx2 = farthest_point_sample(point1, 256, cuda=self.cuda)
        torch.cuda.empty_cache()
        branch2_xyz = index_points(branch1_xyz, idx2, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        branch2_points_FP, normal = self.la3(xyz=branch2_xyz, base_xyz=branch1_xyz, normal=normal,
                                             feature=branch1_points_FP, FPS_idx=idx2)

        idx3 = farthest_point_sample(branch2_points_FP, 128, cuda=self.cuda)
        torch.cuda.empty_cache()
        branch3_xyz = index_points(branch2_xyz, idx3, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        branch3_points_FP, normal = self.la4(xyz=branch3_xyz, base_xyz=branch2_xyz, normal=normal,
                                             feature=branch2_points_FP, FPS_idx=idx3)

        idx4 = farthest_point_sample(branch3_points_FP, 64, cuda=self.cuda)
        torch.cuda.empty_cache()
        branch4_xyz = index_points(branch3_xyz, idx4, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        branch4_points_FP, normal = self.la5(xyz=branch4_xyz, base_xyz=branch3_xyz, normal=normal,
                                             feature=branch3_points_FP, FPS_idx=idx4)



        ####54
        branch4_points, normal = self.upla4(xyz=branch4_xyz, base_xyz=branch4_xyz, normal=normal,
                                             feature=branch4_points_FP, FPS_idx=None)

        branch4_points = self.up5_4(branch3_xyz, branch4_xyz, None, branch4_points)
        branch3_points = branch3_points_FP + branch4_points



        ####43
        branch3_points, normal = self.upla3(xyz=branch3_xyz, base_xyz=branch3_xyz, normal=normal,
                                             feature=branch3_points, FPS_idx=None)

        branch3_points = self.up4_3(branch2_xyz, branch3_xyz, None, branch3_points)
        branch2_points = branch2_points_FP + branch3_points


        #####32
        branch2_points, normal = self.upla2(xyz=branch2_xyz, base_xyz=branch2_xyz, normal=normal,
                                             feature=branch2_points, FPS_idx=None)

        branch2_points = self.up3_2(branch1_xyz, branch2_xyz, None, branch2_points)
        branch1_points = branch1_points_FP + branch2_points


        #####21
        branch1_points, normal = self.upla1(xyz=branch1_xyz, base_xyz=branch1_xyz, normal=normal,
                                             feature=branch1_points, FPS_idx=None)

        points0 = self.up2_1(FPS_xyz0, branch1_xyz, None, branch1_points)
        points0 = points0_FPS + points0


        #####10
        points0, normal = self.upla0(xyz=FPS_xyz0, base_xyz=FPS_xyz0, normal=normal,
                                             feature=points0, FPS_idx=None)

        points = self.up1_0(xyz, FPS_xyz0, None, points0)
        points = points_FPS + points



        # x = torch.cat((x1, x2, x3, x4), dim=2)
        x = self.conv6(points)#.squeeze(-1)
        x = x.max(dim=1, keepdim=True)[0]

        # label = label.view(batch_size, -1, 1, 1)
        label = self.conv7(label)

        x = torch.cat((x, label), dim=2)
        x = x.repeat(1, num_points, 1)

        final = torch.cat((x, points), 2)


        return xyz, final#, ret2, ret3, ret4


# class KeepHighResolutionModuleSemiSeg(nn.Module):
#
#     def __init__(self, data_C, b1_C, b2_C, b3_C, b4_C):
#         super(KeepHighResolutionModulePartSeg, self).__init__()
#
#         self.local_num_neighbors = [16, 32]
#         self.neighbour = 40
#         # self.transform_net = Transform_Net()
#         # self.transform_la = LocalAggregation(3 * 2 + 7, b1_C, [self.neighbour, ])
#
#         # self.fc_start = nn.Linear(3, 8)
#         # self.bn_start = nn.Sequential(
#         #     nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
#         #     nn.LeakyReLU(0.2)
#         # )
#         #
#         # # encoding layers
#         # self.encoder = nn.ModuleList([
#         #     LocalFeatureAggregation(8, 16, num_neighbors=16),
#         #     LocalFeatureAggregation(32, 64, num_neighbors=16),
#         #     LocalFeatureAggregation(128, 128, num_neighbors=16),
#         #     LocalFeatureAggregation(256, 256, num_neighbors=16)
#         # ])
#         #
#         # self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())
#         #
#         # self.decimation = 4
#
#         # self.start = SharedMLP(7, b1_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.start_la1 = LocalAggregation(3 * 2 + 7, b1_C, [self.neighbour, ])
#         self.start_la2 = LocalAggregation(3 * 2 + 7, b2_C, [self.neighbour, ])
#         self.start_la3 = LocalAggregation(3 * 2 + 7, b3_C, [self.neighbour, ])
#         self.start_la4 = LocalAggregation(3 * 2 + 7, b4_C, [self.neighbour, ])
#
#         self.start_fuse = Fuse_Stage(64, 64, 64, 64)
#
#         ###stage2
#         self.stage2_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
#         self.stage2_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
#         self.stage2_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
#         self.stage2_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])
#
#         self.stage2_fuse = Fuse_Stage(64, 64, 64, 64)
#
#         ###stage3
#         self.stage3_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
#         self.stage3_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
#         self.stage3_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
#         self.stage3_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])
#
#         self.stage3_fuse = Fuse_Stage(64, 64, 64, 64)
#
#         ###stage4
#         self.stage4_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
#         self.stage4_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
#         self.stage4_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
#         self.stage4_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])
#
#         self.stage4_fuse = Fuse_Stage(64, 64, 64, 64)
#
#         self.conv6 = SharedMLP(512, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.conv7 = SharedMLP(16, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.final_class = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.final_pool = AttentivePooling(1024, 1024, 256)
#
#         self.conv_x1 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.conv_x2 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x2_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.conv_x3 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x3_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.conv_x4 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         # self.conv_x4_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#         self.final = SharedMLP(512, 512, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#
#     def forward(self, xyz, points=None, label=None):
#
#         batch_size = xyz.size(0)
#         num_points = xyz.size(2)
#
#         branch1_xyz = xyz  ### B C N
#
#         if points is not None:
#             branch1_norm = points
#         else:
#             branch1_norm = None
#
#         # x0 = self.transform_la(branch1_xyz, trans=True)
#         # trans = self.transform_net(x0)
#         # branch1_xyz = torch.bmm(branch1_xyz.transpose(2,1), trans).transpose(2,1)
#
#         # branch2_idx_FP = farthest_point_sample(branch1_xyz, 512)
#         # branch2_xyz_FP = index_points(branch1_xyz, branch2_idx_FP)  # FPS generate branch2
#
#         # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
#         # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B C N
#
#         # branch1_points = self.stage2_1(branch1_xyz.unsqueeze(-1)).squeeze(-1)
#
#         branch1_points = self.start_la1(branch1_xyz, norm=branch1_norm)
#
#         # x1 = self.conv_x1(branch1_points.unsqueeze(-1))
#
#         branch2_idx_FP = farthest_point_sample(branch1_xyz.permute(0, 2, 1).contiguous(), 512)
#         branch2_xyz = index_points(branch1_xyz.permute(0, 2, 1).contiguous(), branch2_idx_FP)
#         if points is not None:
#             branch2_norm = index_points(branch1_norm.permute(0, 2, 1), branch2_idx_FP).permute(0, 2, 1).contiguous()
#         else:
#             branch2_norm = None
#         branch2_points_FP = self.start_la2(branch2_xyz.permute(0, 2, 1).contiguous(),
#                                            norm=branch2_norm)  # FPS generate branch2
#         # branch2_points_FP = self.start_la2(branch2_xyz.permute(0,2,1).contiguous()) # FPS generate branch2
#
#         branch3_idx_FP = farthest_point_sample(branch2_xyz, 256)
#         branch3_xyz = index_points(branch2_xyz, branch3_idx_FP)
#         if points is not None:
#             branch3_norm = index_points(branch2_norm.permute(0, 2, 1), branch3_idx_FP).permute(0, 2, 1).contiguous()
#         else:
#             branch3_norm = None
#         branch3_points_FP = self.start_la3(branch3_xyz.permute(0, 2, 1).contiguous(), norm=branch3_norm)
#
#         branch4_idx_FP = farthest_point_sample(branch3_xyz, 128)
#         branch4_xyz = index_points(branch3_xyz, branch4_idx_FP)
#         if points is not None:
#             branch4_norm = index_points(branch3_norm.permute(0, 2, 1), branch4_idx_FP).permute(0, 2, 1).contiguous()
#         else:
#             branch4_norm = None
#         branch4_points_FP = self.start_la4(branch4_xyz.permute(0, 2, 1).contiguous(), norm=branch4_norm)
#
#         ################ fuse
#         branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.start_fuse(branch1_points,
#                                                                                                   branch2_points_FP,
#                                                                                                   branch3_points_FP,
#                                                                                                   branch4_points_FP,
#                                                                                                   branch2_idx_FP,
#                                                                                                   branch3_idx_FP,
#                                                                                                   branch4_idx_FP)
#
#         x1 = self.conv_x1(branch1_points.unsqueeze(-1).contiguous())
#
#         branch1_xyz = branch1_xyz.permute(0, 2, 1).contiguous()
#
#         ############## Local Aggregation  stage2
#         branch1_points = self.stage2_la1(branch1_points, branch1_xyz, norm=branch1_norm)
#         branch2_points_FP = self.stage2_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm)  # FPS generate branch2
#         branch3_points_FP = self.stage2_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
#         branch4_points_FP = self.stage2_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)
#
#         ##################  Fuse
#         branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage2_fuse(branch1_points,
#                                                                                                    branch2_points_FP,
#                                                                                                    branch3_points_FP,
#                                                                                                    branch4_points_FP,
#                                                                                                    branch2_idx_FP,
#                                                                                                    branch3_idx_FP,
#                                                                                                    branch4_idx_FP)
#
#         x2 = self.conv_x2(branch1_points.unsqueeze(-1).contiguous())
#
#         ############## Local Aggregation  stage3
#         branch1_points = self.stage3_la1(branch1_points, branch1_xyz, norm=branch1_norm)
#         branch2_points_FP = self.stage3_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm)  # FPS generate branch2
#         branch3_points_FP = self.stage3_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
#         branch4_points_FP = self.stage3_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)
#
#         ##################  Fuse
#         branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage3_fuse(branch1_points,
#                                                                                                    branch2_points_FP,
#                                                                                                    branch3_points_FP,
#                                                                                                    branch4_points_FP,
#                                                                                                    branch2_idx_FP,
#                                                                                                    branch3_idx_FP,
#                                                                                                    branch4_idx_FP)
#
#         x3 = self.conv_x3(branch1_points.unsqueeze(-1).contiguous())
#
#         ############## Local Aggregation  stage4
#         branch1_points = self.stage4_la1(branch1_points, branch1_xyz, norm=branch1_norm)
#         branch2_points_FP = self.stage4_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm)  # FPS generate branch2
#         branch3_points_FP = self.stage4_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
#         branch4_points_FP = self.stage4_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)
#
#         ##################  Fuse
#         branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage4_fuse(branch1_points,
#                                                                                                    branch2_points_FP,
#                                                                                                    branch3_points_FP,
#                                                                                                    branch4_points_FP,
#                                                                                                    branch2_idx_FP,
#                                                                                                    branch3_idx_FP,
#                                                                                                    branch4_idx_FP)
#
#         x4 = self.conv_x4(branch1_points.unsqueeze(-1).contiguous())
#
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#         x = self.conv6(x).squeeze(-1)
#         x = x.max(dim=-1, keepdim=True)[0]
#
#         # label = label.view(batch_size, -1, 1, 1)
#         # label = self.conv7(label).squeeze(-1)
#
#         # x = torch.cat((x, label), dim=1)
#         x = x.repeat(1, 1, num_points).unsqueeze(-1)
#
#         final = torch.cat((x, x1, x2, x3, x4), 1)
#
#         return xyz, final


#########################################################


# class PointNetSetAbstraction(nn.Module):
#     def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
#         super(PointNetSetAbstraction, self).__init__()
#         self.npoint = npoint
#         self.radius = radius
#         self.nsample = nsample
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm2d(out_channel))
#             last_channel = out_channel
#         self.group_all = group_all
#
#     def forward(self, xyz, points):
#         """
#         Input:
#             xyz: input points position data, [B, C, N]
#             points: input points data, [B, D, N]
#         Return:
#             new_xyz: sampled points position data, [B, C, S]
#             new_points_concat: sample points feature data, [B, D', S]
#         """
#         xyz = xyz.permute(0, 2, 1)
#         if points is not None:
#             points = points.permute(0, 2, 1)
#
#         if self.group_all:
#             new_xyz, new_points = sample_and_group_all(xyz, points)
#         else:
#             new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
#         # new_xyz: sampled points position data, [B, npoint, C]
#         # new_points: sampled points data, [B, npoint, nsample, C+D]
#         new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.relu(bn(conv(new_points)))
#
#         new_points = torch.max(new_points, 2)[0]
#         new_xyz = new_xyz.permute(0, 2, 1)
#         return new_xyz, new_points
#
#
# class PointNetSetAbstractionMsgTest(nn.Module):  ##################################
#     def __init__(self, radiusQuad, inputQuad, outputQuad, npoint, radius_list, nsample_list, in_channel, mlp_list):
#         super(PointNetSetAbstractionMsgTest, self).__init__()
#         self.npoint = npoint
#
#         self.radiusQuad = radiusQuad  #######################
#         # self.outputQuad = outputQuad  ###################
#
#         self.radius_list = radius_list
#         self.nsample_list = nsample_list
#         self.conv_blocks = nn.ModuleList()
#         self.bn_blocks = nn.ModuleList()
#         for i in range(len(mlp_list)):
#             convs = nn.ModuleList()
#             bns = nn.ModuleList()
#             last_channel = in_channel + 32  #############################
#             for out_channel in mlp_list[i]:
#                 convs.append(nn.Conv2d(last_channel, out_channel, 1))
#                 bns.append(nn.BatchNorm2d(out_channel))
#                 last_channel = out_channel
#             self.conv_blocks.append(convs)
#             self.bn_blocks.append(bns)
#
#         self.conv1 = nn.Sequential(
#             conv_bn(inputQuad, outputQuad, [1, 2], [1, 2]),
#             conv_bn(outputQuad, outputQuad, [1, 2], [1, 2]),
#             conv_bn(outputQuad, outputQuad, [1, 2], [1, 2])
#         )
#
#     def forward(self, xyz, points):
#         """
#         Input:
#             xyz: input points position data, [B, C, N]
#             points: input points data, [B, D, N]
#         Return:
#             new_xyz: sampled points position data, [B, C, S]
#             new_points_concat: sample points feature data, [B, D', S]
#         # """
#         xyz = xyz.permute(0, 2, 1)
#         if points is not None:
#             points = points.permute(0, 2, 1)
#
#         B, N, C = xyz.shape
#
#         a = torch.rand([1, 5, 3])
#         sqrdists = square_distance(a, a)
#
#         maskT = sqrdists < 0.5
#
#         idxtemp = torch.arange(5).repeat(5, 1).contiguous().repeat(1, 1, 1)
#
#         idx = torch.tensor(999).repeat(1, 5, 5)
#
#         idx[maskT] = idxtemp[maskT]
#
#         _, group_idx = torch.topk(sqrdists, 2, dim=-1, largest=False, sorted=False)
#
#         grouped_xyz = index_points(a, group_idx)
#
#         ###################################################################### 8 quadrant
#         radiusQuad = self.radiusQuad
#         _, grouped_pointsQuad, grouped_edgeQuad, idx = pointsift_group(radiusQuad, xyz, points)
#
#         grouped_pointsQuad = grouped_pointsQuad.permute(0, 3, 1, 2).contiguous()
#         new_pointsQuad = self.conv1(grouped_pointsQuad)
#         new_pointsQuad = new_pointsQuad.squeeze(-1).permute(0, 2, 1).contiguous()
#
#         ######################################################################
#         S = self.npoint
#         new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
#
#         # group_idx = query_ball_point(0.1, 32, xyz, new_xyz)  ############################
#         # grouped_points = index_points(new_points, group_idx)  #############################
#         # new_points = torch.max(grouped_points, 2)[0]   ############################
#
#         new_points_list = []
#         for i, radius in enumerate(self.radius_list):
#             K = self.nsample_list[i]
#             group_idx = query_ball_point(radius, K, xyz, new_xyz)
#
#             grouped_points = index_points(new_pointsQuad, group_idx)  #############################
#             # grouped_xyz = index_points(xyz, group_idx)   ###################%%%%%%%%%%%%
#             # grouped_xyz -= new_xyz.view(B, S, 1, C)    ##############%%%%%%%%%%%%%%%%%
#             #
#             # if points is not None:
#             #     grouped_points = index_points(points, group_idx)
#             #     grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
#             # else:
#             #     grouped_points = grouped_xyz
#
#             grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
#             for j in range(len(self.conv_blocks[i])):
#                 conv = self.conv_blocks[i][j]
#                 bn = self.bn_blocks[i][j]
#                 grouped_points = F.relu(bn(conv(grouped_points)))
#             new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
#             new_points_list.append(new_points)
#
#         new_xyz = new_xyz.permute(0, 2, 1)
#         new_points_concat = torch.cat(new_points_list, dim=1)
#         return new_xyz, new_points_concat
#
#
# class PointNetSetAbstractionMsg(nn.Module):
#     def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
#         super(PointNetSetAbstractionMsg, self).__init__()
#         self.npoint = npoint
#         self.radius_list = radius_list
#         self.nsample_list = nsample_list
#         self.conv_blocks = nn.ModuleList()
#         self.bn_blocks = nn.ModuleList()
#
#         for i in range(len(mlp_list)):
#             convs = nn.ModuleList()
#             bns = nn.ModuleList()
#             last_channel = in_channel + 3
#             for out_channel in mlp_list[i]:
#                 convs.append(nn.Conv2d(last_channel, out_channel, 1))
#                 bns.append(nn.BatchNorm2d(out_channel))
#                 last_channel = out_channel
#             self.conv_blocks.append(convs)
#             self.bn_blocks.append(bns)
#
#     def forward(self, xyz, points):
#         """
#         Input:
#             xyz: input points position data, [B, C, N]
#             points: input points data, [B, D, N]
#         Return:
#             new_xyz: sampled points position data, [B, C, S]
#             new_points_concat: sample points feature data, [B, D', S]
#         # """
#
#         xyz = xyz.permute(0, 2, 1)
#         if points is not None:
#             points = points.permute(0, 2, 1)
#
#         B, N, C = xyz.shape
#         S = self.npoint
#         new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
#         new_points_list = []
#         for i, radius in enumerate(self.radius_list):
#             K = self.nsample_list[i]
#             group_idx = query_ball_point(radius, K, xyz, new_xyz)
#             grouped_xyz = index_points(xyz, group_idx)
#             grouped_xyz -= new_xyz.view(B, S, 1, C)
#             if points is not None:
#                 grouped_points = index_points(points, group_idx)
#                 grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
#             else:
#                 grouped_points = grouped_xyz
#
#             grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
#             for j in range(len(self.conv_blocks[i])):
#                 conv = self.conv_blocks[i][j]
#                 bn = self.bn_blocks[i][j]
#                 grouped_points = F.relu(bn(conv(grouped_points)))
#             new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
#             new_points_list.append(new_points)
#
#         new_xyz = new_xyz.permute(0, 2, 1)
#         new_points_concat = torch.cat(new_points_list, dim=1)
#         return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, act=False, cuda=False):
        super(PointNetFeaturePropagation, self).__init__()

        self.cuda = cuda

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.act = act

        # self.final = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))  #####################################
        # self.conv = SharedMLP(in_channel, out_channel, bn=True, activation_fn=False)  #####################################
        self.conv = Linear(in_channel, out_channel, bn=False, act=self.act)  #####################################

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)



        # points2 = points2.permute(0, 2, 1).contiguous()
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx.int(), cuda=self.cuda, is_group=True) * weight.view(B, N, 3, 1), dim=2)

        new_points = interpolated_points  ##########################
        new_points = self.conv(new_points)  ######################################

        # if points1 is not None:
        #     points1 = points1.permute(0, 2, 1)
        #     new_points = torch.cat([points1, interpolated_points], dim=-1)
        # else:
        #     new_points = interpolated_points
        #
        # new_points = new_points.permute(0, 2, 1)
        #
        # # new_points = self.final(new_points.unsqueeze(-1)).squeeze(-1)   ############################
        #
        #
        #
        # for i, conv in enumerate(self.mlp_convs):
        #     bn = self.mlp_bns[i]
        #     # new_points = F.relu(bn(conv(new_points)))
        #     new_points = bn(conv(new_points))  ###############################
        return new_points

# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp, trans_channel):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2)    ##########################################
#         self.trans_conv = SharedMLP(trans_channel, mlp[0], bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))  #######################
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel
#
#     def forward(self, xyz1, xyz2, points1, points2):
#         """
#         Input:
#             xyz1: input points position data, [B, C, N]
#             xyz2: sampled input points position data, [B, C, S]
#             points1: input points data, [B, D, N]
#             points2: input points data, [B, D, S]
#         Return:
#             new_points: upsampled points data, [B, D', N]
#         """
#
#         # xyz1 = xyz1.permute(0, 2, 1)
#         # xyz2 = xyz2.permute(0, 2, 1)
#         #
#         # points2 = points2.permute(0, 2, 1)
#         # B, N, C = xyz1.shape
#         # _, S, _ = xyz2.shape
#         #
#         # if S == 1:
#         #     interpolated_points = points2.repeat(1, N, 1)
#         # else:
#         #     dists = square_distance(xyz1, xyz2)
#         #     dists, idx = dists.sort(dim=-1)
#         #     dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
#         #
#         #     dist_recip = 1.0 / (dists + 1e-8)
#         #     norm = torch.sum(dist_recip, dim=2, keepdim=True)
#         #     weight = dist_recip / norm
#         #     # a = index_points(points2, idx)   #########################
#         #     interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
#         #
#         # if points1 is not None:
#         #     points1 = points1.permute(0, 2, 1)
#         #     new_points = torch.cat([points1, interpolated_points], dim=-1)
#         # else:
#         #     new_points = interpolated_points
#         #
#         # new_points = new_points.permute(0, 2, 1)
#         # for i, conv in enumerate(self.mlp_convs):
#         #     bn = self.mlp_bns[i]
#         #     # new_points = F.relu(bn(conv(new_points)))
#         #     new_points = self.lrelu(bn(conv(new_points)))
#         # return new_points
#
#
#         #############################################################################################
#
#
#         points2 = self.trans_conv(points2.unsqueeze(-1).contiguous()).squeeze(-1)
#         inner = torch.matmul(points1.transpose(1, 2).contiguous(), points2)
#         s_norm_2 = torch.sum(points2 ** 2, dim=1)  # (bs, v2)
#         t_norm_2 = torch.sum(points1 ** 2, dim=1)  # (bs, v1)
#         d_norm_2 = s_norm_2.unsqueeze(1).contiguous() + t_norm_2.unsqueeze(2).contiguous() - 2 * inner
#
#         nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
#
#         batch_size, num_points, k = nearest_index.size()
#
#         id_0 = torch.arange(batch_size).view(-1, 1, 1)
#
#         # points2 = points2.transpose(2, 1).contiguous()  # (bs, num_points, num_dims)
#         points2 = points2.permute(0,2,1).contiguous()
#         feature = points2[id_0, nearest_index]  # (bs, num_points, k, num_dims)
#         feature = feature.permute(0, 1, 3, 2).squeeze(-1).contiguous()
#
#
#
#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1).contiguous()
#             feature = torch.cat([points1, feature], dim=-1)
#
#
#
#         feature = feature.permute(0, 2, 1).contiguous()
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             feature = self.lrelu(bn(conv(feature)))
#         return feature

