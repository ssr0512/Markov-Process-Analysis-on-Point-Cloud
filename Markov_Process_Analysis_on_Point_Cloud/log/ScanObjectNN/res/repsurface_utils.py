"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.pointnet2_utils import query_knn_point, query_ball_point#, farthest_point_sample, index_points
from modules.polar_utils import xyz2sphere
from modules.recons_utils import cal_const, cal_normal, cal_center, check_nan_umb
import numpy as np

def sample_and_group(npoint, radius, nsample, center, normal, feature, return_normal=True, return_polar=False, cuda=False):
    """
    Input:
        center: input points position data
        normal: input points normal data
        feature: input points feature
    Return:
        new_center: sampled points position data
        new_normal: sampled points normal data
        new_feature: sampled points feature
    """
    # sample
    fps_idx = farthest_point_sample(center, npoint)#, cuda=cuda)  # [B, npoint, A]
    torch.cuda.empty_cache()
    # sample center
    new_center = index_points(center, fps_idx)#, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()
    # sample normal
    new_normal = index_points(normal, fps_idx)#, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()

    # group
    idx = query_ball_point(radius, nsample, center, new_center, cuda=cuda)
    torch.cuda.empty_cache()
    # group normal
    group_normal = index_points(normal, idx)#, cuda=cuda, is_group=True)  # [B, npoint, nsample, B]
    torch.cuda.empty_cache()
    # group center
    group_center = index_points(center, idx)#, cuda=cuda, is_group=True)  # [B, npoint, nsample, A]
    torch.cuda.empty_cache()
    group_center_norm = group_center - new_center.unsqueeze(2)
    torch.cuda.empty_cache()

    # group polar
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)
    if feature is not None:
        group_feature = index_points(feature, idx)#, cuda=cuda, is_group=True)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1) if return_normal \
            else torch.cat([group_center_norm, group_feature], dim=-1)
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1)

    return new_center, new_normal, new_feature

def sample_and_group_all(center, normal, feature, return_normal=True, return_polar=False):
    """
    Input:
        center: input centroid position data
        normal: input normal data
        feature: input feature data
    Return:
        new_center: sampled points position data
        new_normal: sampled points position data
        new_feature: sampled points data
    """
    device = center.device
    B, N, C = normal.shape

    new_center = torch.zeros(B, 1, 3).to(device)
    new_normal = new_center

    group_normal = normal.view(B, 1, N, C)
    group_center = center.view(B, 1, N, 3)
    if return_polar:
        group_polar = xyz2sphere(group_center)
        group_center = torch.cat([group_center, group_polar], dim=-1)

    new_feature = torch.cat([group_center, group_normal, feature.view(B, 1, N, -1)], dim=-1) if return_normal \
        else torch.cat([group_center, feature.view(B, 1, N, -1)], dim=-1)

    return new_center, new_normal, new_feature

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
    group_xyz = index_points(xyz, idx)[:, :, 1:]#, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
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

################################################################
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest

        # a = xyz[batch_indices,:,:]       ###########################
        # b = xyz[batch_indices,farthest,:]   #######################
        # centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)   ################%%%%%%%%%%%%%%%%%%
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)  ######################################
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    dist, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    return dist, group_idx

class SurfaceAbstraction(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, return_polar=True, return_normal=True, cuda=False):
        super(SurfaceAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all:
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_polar=self.return_polar,
                                                                       return_normal=self.return_normal)
        else:
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_polar=self.return_polar,
                                                                   return_normal=self.return_normal, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature

class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, radius, nsample, feat_channel, pos_channel, mlp, group_all,
                 return_normal=True, return_polar=False, cuda=False):
        super(SurfaceAbstractionCD, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel
        self.group_all = group_all

        self.mlp_l0 = nn.Conv2d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv2d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm2d(mlp[0])
        self.bn_f0 = nn.BatchNorm2d(mlp[0])

        # mlp_l0+mlp_f0 can be considered as the first layer of mlp_convs
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all:
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_normal=self.return_normal,
                                                                       return_polar=self.return_polar)
        else:
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_normal=self.return_normal,
                                                                   return_polar=self.return_polar, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1)

        # init layer
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
        new_feature = loc + feat
        new_feature = F.relu(new_feature)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature

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

############################################################################################

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
    def __init__(self, in_channels, out_channels, knn, usetanh=False, residual=False):
        super(LocalMerge, self).__init__()

        self.knn = knn
        self.usetanh = usetanh
        self.residual = residual
        self.fc1 = Linear(out_channels*2, out_channels, bn=False)
        self.fc2 = Linear(out_channels*2, out_channels, bn=False)
        # self.conv_normal = nn.Linear(10, out_channels)
        # self.conv_normal2 = nn.Linear(10, out_channels)

        self.xyz_Trans = LocalTrans(3, out_channels, knn, usetanh=self.usetanh, residual=True)
        self.normal_Trans = LocalTrans(10, out_channels, knn, usetanh=self.usetanh, residual=True)
        self.feature_Trans = LocalTrans(in_channels, out_channels, knn, usetanh=self.usetanh, residual=self.residual)
        self.feature_Trans2 = LocalTrans(in_channels, out_channels, knn, usetanh=self.usetanh, residual=self.residual)

    def forward(self, xyz, base_xyz, normal=None, feature=None, FPS_idx=None, xyz_flag=True):

        # if new_offset is None:
        #     new_offset = offset

        # # idx, dist = query_knn_point(self.knn, base_xyz, xyz, cuda=True)  ### B N K
        # # idx, _ = pointops.knnquery(self.knn, base_xyz, xyz, offset, new_offset)  # [M, nsample]
        # dist, idx = knn_point(self.knn, base_xyz, xyz)  ### B N K


        # # torch.cuda.empty_cache()

        # if feature is None and FPS_idx is None and xyz_flag is True:
        #     dist, idx = knn_point(self.knn, base_xyz, xyz)  ### B N K
        #     idx = idx.int()
        # elif feature is not None and FPS_idx is not None and xyz_flag is True:
        #     fs = index_points(feature, FPS_idx, cuda=True, is_group=False)
        #     dist, idx = knn_point(self.knn, feature, fs)  ### B N K
        #     idx = idx.int()
        # elif feature is not None and FPS_idx is not None and xyz_flag is False:
        #     dist, idx = knn_point(self.knn, base_xyz, xyz)  ### B N K
        #     idx = idx.int()
        # elif feature is not None and FPS_idx is None:
        #     dist, idx = knn_point(self.knn, feature, feature)  ### B N K
        #     idx = idx.int()



        if feature is None and FPS_idx is None:
            dist, idx = knn_point(self.knn, base_xyz, xyz)  ### B N K
        elif feature is not None and FPS_idx is None:
            dist, idx = knn_point(self.knn, base_xyz, xyz)  ### B N K
            _, idx_feature = knn_point(self.knn, feature, feature)  ### B N K
        else:
            dist, idx = knn_point(self.knn, base_xyz, xyz)  ### B N K
            fs = index_points(feature, FPS_idx)
            _, idx_feature = knn_point(self.knn, feature, fs)  ### B N K



        if feature is None:
            xyz_f = self.xyz_Trans(features=xyz, idx=idx, pos=base_xyz, FPS_idx=FPS_idx, xyz=True)
            # normal_f = self.normal_Trans(features=normal, idx=idx, pos=base_xyz, FPS_idx=FPS_idx)
            merge_features = xyz_f
            # merge_features = self.fc1(torch.cat((xyz_f, normal_f), dim=2))
        else:
            # xyz_f = self.xyz_Trans(features=base_xyz, idx=idx, pos=base_xyz, FPS_idx=FPS_idx, xyz=True)
            # normal_f = self.normal_Trans(features=normal, idx=idx, pos=base_xyz, FPS_idx=FPS_idx)
            merge_features1 = self.feature_Trans(features=feature, idx=idx, pos=base_xyz, FPS_idx=FPS_idx)
            merge_features2 = self.feature_Trans2(features=feature, idx=idx_feature, pos=base_xyz, FPS_idx=FPS_idx)
            merge_features = self.fc2(torch.cat((merge_features1, merge_features2), dim=2))
            # merge_features = self.fc2(torch.cat((xyz_f, normal_f, features_pos, features_content), dim=2))

        # if FPS_idx is not None:
        #     normal = index_points(normal, FPS_idx)
        #     # normal = normal[FPS_idx.long(), :]


        return merge_features, normal, idx, dist

class LocalTrans(nn.Module):
    def __init__(self, in_c, out_c, patch_num, usetanh=False, residual=False):
        super(LocalTrans, self).__init__()

        self.patchNum = patch_num
        self.residual = residual
        self.usetanh = usetanh

        self.out_c = out_c

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
            # residual = index_points(features, FPS_idx, cuda=self.cuda, is_group=False)
            # torch.cuda.empty_cache()
            # pos_emb = index_points(pos, FPS_idx, cuda=self.cuda, is_group=False)
            # torch.cuda.empty_cache()
            # residual = features[FPS_idx.long(), :]
            # pos_emb = pos[FPS_idx.long(), :]
            residual = index_points(features, FPS_idx)
            # pos_emb = index_points(pos, FPS_idx)
            # torch.cuda.empty_cache()


            center_features = residual

        else:
            residual = features
            # pos_emb = pos

            # features = self.fc_pre(features)
            center_features = features



        if self.residual is True:
            residual = self.conv_res(residual)

        # knn_xyz = index_points(pos, idx, cuda=self.cuda, is_group=True)
        # torch.cuda.empty_cache()
        # knn_N, knn_C = idx.shape[0], pos.shape[1]
        # knn_xyz = pos[idx.view(-1).long(), :].view(knn_N, self.patchNum, knn_C)
        # knn_xyz = index_points(pos, idx)
        # torch.cuda.empty_cache()

        if xyz is True:

            local_query = self.q(center_features).unsqueeze(-2)



            local_key = index_points(features, idx)
            # torch.cuda.empty_cache()
            local_key = local_key - center_features.unsqueeze(-2)
            local_key = self.k(local_key)#.permute(0,1,3,2).contiguous()


            local_value = index_points(features, idx)
            # torch.cuda.empty_cache()
            local_value = local_value - center_features.unsqueeze(-2)
            local_value = self.v(local_value)


            energy = local_query - local_key #+ pos_emb


            if self.usetanh is True:
                attention = self.tanh(energy) / self.patchNum
                context = torch.matmul(attention, local_value).squeeze(-2) #+ anchor_value  ###nn.bmm
            else:
                attention = F.softmax(energy/np.sqrt(local_key.size(-1)), dim=-2)

                offset = torch.sum(attention, dim=2, keepdim=True)
                attention = attention - offset
                # context = torch.matmul(attention, local_value).squeeze(-2)  ###nn.bmm
                # context = torch.einsum('bnkc,bnkc->bnc', attention, local_value + pos_emb)
                context = attention * (local_value)# + pos_emb)
                context = torch.max(context, 2)[0]

            # ##############################
            # local_query1 = self.q1(center_features).unsqueeze(-2)
            #
            #
            # local_key1 = index_points(features, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            # local_key1 = local_key1 - center_features.unsqueeze(-2)
            # local_key1 = self.k1(local_key1)#.permute(0,1,3,2).contiguous()
            #
            #
            # local_value1 = index_points(features, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            # local_value1 = local_value1 - center_features.unsqueeze(-2)
            # local_value1 = self.v1(local_value1)
            #
            #
            # energy1 = local_query1 - local_key1 #+ pos_emb
            #
            #
            # attention1 = F.softmax(energy1/np.sqrt(local_key1.size(-1)), dim=-2)
            #
            # offset1 = torch.sum(attention1, dim=2, keepdim=True)
            # attention1 = attention1 - offset1
            # context1 = attention1 * (local_value1)# + pos_emb)
            # context1 = torch.max(context1, 2)[0]
            #
            # ################################
            # local_query2 = self.q2(center_features).unsqueeze(-2)
            #
            #
            # local_key2 = index_points(features, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            # local_key2 = local_key2 - center_features.unsqueeze(-2)
            # local_key2 = self.k2(local_key2)#.permute(0,1,3,2).contiguous()
            #
            #
            # local_value2 = index_points(features, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            # local_value2 = local_value2 - center_features.unsqueeze(-2)
            # local_value2 = self.v2(local_value2)
            #
            #
            # energy2 = local_query2 - local_key2 #+ pos_emb
            #
            #
            # attention2 = F.softmax(energy2/np.sqrt(local_key2.size(-1)), dim=-2)
            #
            # offset2 = torch.sum(attention2, dim=2, keepdim=True)
            # attention2 = attention2 - offset2
            # context2 = attention2 * (local_value2)# + pos_emb)
            # context2 = torch.max(context2, 2)[0]
            #
            #
            # #############################################
            # local_query3 = self.q3(center_features).unsqueeze(-2)
            #
            #
            # local_key3 = index_points(features, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            # local_key3 = local_key3 - center_features.unsqueeze(-2)
            # local_key3 = self.k3(local_key3)#.permute(0,1,3,2).contiguous()
            #
            #
            # local_value3 = index_points(features, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            # local_value3 = local_value3 - center_features.unsqueeze(-2)
            # local_value3 = self.v3(local_value3)
            #
            #
            # energy3 = local_query3 - local_key3 #+ pos_emb
            #
            #
            # attention3 = F.softmax(energy3/np.sqrt(local_key3.size(-1)), dim=-2)
            #
            # offset3 = torch.sum(attention3, dim=2, keepdim=True)
            # attention3 = attention3 - offset3
            # context3 = attention3 * (local_value3)# + pos_emb)
            # context3 = torch.max(context3, 2)[0]


        else:
            local_query = self.q(center_features).unsqueeze(-2)

            # N, C = idx.shape[0], self.out_c

            local_key = self.k(features)
            local_key = index_points(local_key, idx)
            # torch.cuda.empty_cache()


            local_value = self.v(features)
            local_value = index_points(local_value, idx)
            # torch.cuda.empty_cache()

            energy = local_query - local_key #+ pos_emb


            if self.usetanh is True:
                attention = self.tanh(energy) / self.patchNum
                context = torch.matmul(attention, local_value).squeeze(-2) #+ anchor_value  ###nn.bmm
            else:
                attention = F.softmax(energy/np.sqrt(local_key.size(-1)), dim=-2)
                offset = torch.sum(attention, dim=2, keepdim=True)
                attention = attention - offset
                # context = torch.matmul(attention, local_value).squeeze(-2)  ###nn.bmm
                # context = torch.einsum('bnkc,bnkc->bnc', attention, local_value + pos_emb)
                context = attention * (local_value)# + pos_emb)
                context = torch.max(context, 2)[0]

            # ###########################################
            # local_query1 = self.q1(center_features).unsqueeze(-2)
            #
            # local_key1 = self.k1(features)
            # local_key1 = index_points(local_key1, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            #
            #
            # local_value1 = self.v1(features)
            # local_value1 = index_points(local_value1, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            #
            # energy1 = local_query1 - local_key1 #+ pos_emb
            #
            #
            # attention1 = F.softmax(energy1/np.sqrt(local_key1.size(-1)), dim=-2)
            # offset1 = torch.sum(attention1, dim=2, keepdim=True)
            # attention1 = attention1 - offset1
            # context1 = attention1 * (local_value1)# + pos_emb)
            # context1 = torch.max(context1, 2)[0]
            #
            # ###########################################
            # local_query2 = self.q2(center_features).unsqueeze(-2)
            #
            # local_key2 = self.k2(features)
            # local_key2 = index_points(local_key2, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            #
            #
            # local_value2 = self.v2(features)
            # local_value2 = index_points(local_value2, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            #
            # energy2 = local_query2 - local_key2 #+ pos_emb
            #
            #
            # attention2 = F.softmax(energy2/np.sqrt(local_key2.size(-1)), dim=-2)
            # offset2 = torch.sum(attention2, dim=2, keepdim=True)
            # attention2 = attention2 - offset2
            # context2 = attention2 * (local_value2)# + pos_emb)
            # context2 = torch.max(context2, 2)[0]
            #
            #
            # ###########################################
            # local_query3 = self.q3(center_features).unsqueeze(-2)
            #
            # local_key3 = self.k3(features)
            # local_key3 = index_points(local_key3, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            #
            #
            # local_value3 = self.v3(features)
            # local_value3 = index_points(local_value3, idx, cuda=True, is_group=True)
            # # torch.cuda.empty_cache()
            #
            # energy3 = local_query3 - local_key3 #+ pos_emb
            #
            #
            # attention3 = F.softmax(energy1/np.sqrt(local_key1.size(-1)), dim=-2)
            # offset3 = torch.sum(attention1, dim=2, keepdim=True)
            # attention3 = attention3 - offset1
            # context3 = attention3 * (local_value3)# + pos_emb)
            # context3 = torch.max(context3, 2)[0]

        # context = residual + self.ffn(residual - context)
        context = residual + self.ffn(context)

        return context

class KeepHighResolutionModule(nn.Module):

    def __init__(self, data_C, b1_C, b2_C, b3_C, b4_C, cuda=False):
        super(KeepHighResolutionModule, self).__init__()

        self.cuda = cuda
        # self.local_num_neighbors = [8, 32]
        # self.neighbour = 8

        self.drop = nn.Dropout(0.5)  ########################


        self.la0 = LocalMerge(32, 64, 8, usetanh=False, residual=True)
        self.la1 = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        self.la2 = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        self.la3 = LocalMerge(64, 128, 8, usetanh=False, residual=True)
        self.la4 = LocalMerge(128, 256, 8, usetanh=False, residual=True)
        self.la5 = LocalMerge(256, 512, 8, usetanh=False, residual=True)

        self.start = Linear(3, 32, bn=False)

        self.conv3 = Linear(512, 512, bn=False)
        self.conv4 = Linear(512, 1024, bn=False)

        self.final = Linear(512, 1024, bn=False)
        self.final_class = nn.Linear(2048, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)


    def forward(self, xyz, normal):


        # points = self.start1(features=xyz, norm=branch1_norm)
        # points = self.start1(features.permute(0, 2, 1).contiguous())

        xyz = xyz.permute(0, 2, 1).contiguous()
        normal = normal.permute(0, 2, 1).contiguous()

        # start = self.start(xyz)

        points, normal0, knn_idx0, dist0 = self.la0(xyz=xyz, base_xyz=xyz, normal=normal, xyz_flag=True)



        # value0, idx0 = self.pool0(xyz.permute(0,2,1), points)
        # FPS_xyz0 = index_points(xyz.permute(0,2,1), idx0)
        # _, _, c1 = points.shape
        FPS_idx0 = farthest_point_sample(xyz, 512)#, cuda=self.cuda)
        torch.cuda.empty_cache()
        FPS_xyz0 = index_points(xyz, FPS_idx0)#, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        # FPS_xyz0, _ = self.FPS_offset1(xyz.permute(0, 2, 1), points, FPS_idx0, FPS_xyz0_gt)
        # FPS_xyz0, FPS_idx0 = self.FPS_offset1(xyz.permute(0,2,1), points)
        # d1 = self.diffusion1(FPS_xyz0, xyz.permute(0,2,1), FPS_idx0)
        # d1_idx = FPS_idx0


        # points0 = self.Trans0(xyz=FPS_xyz0, base_xyz=xyz.permute(0, 2, 1).contiguous(), features=points, FPS_idx=FPS_idx0)
        points0, normal1, knn_idx1, dist1 = self.la1(xyz=FPS_xyz0, base_xyz=xyz, normal=normal0, feature=points,
                                                     FPS_idx=FPS_idx0, xyz_flag=True)

        # umber_feature0 = index_points(features.permute(0, 2, 1).contiguous(), FPS_idx0, cuda=self.cuda, is_group=False)
        # torch.cuda.empty_cache()
        # points0 = self.lrelu(points0 + self.start2(umber_feature0))

        # _, _, c2 = points0.shape
        FPS_idx1 = farthest_point_sample(FPS_xyz0, 256)#, cuda=self.cuda)
        torch.cuda.empty_cache()
        FPS_xyz1 = index_points(FPS_xyz0, FPS_idx1)#, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        # FPS_xyz1, _ = self.FPS_offset2(FPS_xyz0_gt, points0, FPS_idx1, FPS_xyz1_gt)
        # FPS_xyz1, FPS_idx1 = self.FPS_offset2(FPS_xyz0, points0)
        # d2 = self.diffusion2(FPS_xyz1, FPS_xyz0, FPS_idx1)
        # d2_idx = FPS_idx1

        # point1, normal = self.Trans1(xyz=FPS_xyz1, base_xyz=FPS_xyz0, features=points0, FPS_idx=FPS_idx1)
        point1, normal2, knn_idx2, dist2 = self.la2(xyz=FPS_xyz1, base_xyz=FPS_xyz0, normal=normal1, feature=points0,
                                                    FPS_idx=FPS_idx1, xyz_flag=False)
        # umber_feature1 = index_points(umber_feature0, FPS_idx1, cuda=self.cuda, is_group=False)
        # torch.cuda.empty_cache()
        # point1 = self.lrelu(point1 + self.start3(umber_feature1))

        # branch1_xyz = FPS_xyz1
        # branch1_points = point1

        # _, _, c3 = point1.shape
        idx2 = farthest_point_sample(FPS_xyz1, 128)#, cuda=self.cuda)
        torch.cuda.empty_cache()
        branch2_xyz = index_points(FPS_xyz1, idx2)#, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        branch2_points_FP, normal3, knn_idx3, dist3 = self.la3(xyz=branch2_xyz, base_xyz=FPS_xyz1, normal=normal2,
                                                               feature=point1, FPS_idx=idx2, xyz_flag=True)

        # _, _, c4 = branch2_points_FP.shape
        idx3 = farthest_point_sample(branch2_xyz, 64)#, cuda=self.cuda)
        torch.cuda.empty_cache()
        branch3_xyz = index_points(branch2_xyz, idx3)#, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        branch3_points_FP, normal4, knn_idx4, dist4 = self.la4(xyz=branch3_xyz, base_xyz=branch2_xyz, normal=normal3,
                                                               feature=branch2_points_FP, FPS_idx=idx3, xyz_flag=False)

        # _, _, c5 = branch3_points_FP.shape
        idx4 = farthest_point_sample(branch3_xyz, 32)#, cuda=self.cuda)
        torch.cuda.empty_cache()
        branch4_xyz = index_points(branch3_xyz, idx4)#, cuda=self.cuda, is_group=False)
        torch.cuda.empty_cache()

        branch4_points_FP, normal5, knn_idx5, dist5 = self.la5(xyz=branch4_xyz, base_xyz=branch3_xyz, normal=normal4,
                                                               feature=branch3_points_FP, FPS_idx=idx4, xyz_flag=False)

        final = self.conv3(branch4_points_FP)
        final = self.conv4(final).permute(0, 2, 1).contiguous()

        x1 = F.adaptive_max_pool1d(final, 1)
        x2 = F.adaptive_avg_pool1d(final, 1)
        final_fuse = torch.cat((x1, x2), 1).squeeze(-1)

        final_fuse = self.lrelu(self.bn(self.final_class(final_fuse)))
        # final_fuse = self.drop(final_fuse)

        return final_fuse#, ret2, ret3, ret4

# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp, cuda=False):
#         super(PointNetFeaturePropagation, self).__init__()
#
#         self.cuda = cuda        #################################
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel
#
#         # self.final = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))  #####################################
#         # self.conv = SharedMLP(in_channel, out_channel, bn=True, activation_fn=False)  #####################################
#         self.conv = Linear(in_channel, out_channel, bn=False, act=False)  #####################################
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
#         # xyz1 = xyz1.permute(0, 2, 1)
#         # xyz2 = xyz2.permute(0, 2, 1)
#
#         points2 = self.conv(points2)  ######################################
#
#         # points2 = points2.permute(0, 2, 1).contiguous()
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape
#
#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             dists = square_distance(xyz1, xyz2)
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
#
#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             if self.cuda is True:          ############################
#                 idx = idx.to(torch.int32)  ######################
#
#             interpolated_points = torch.sum(index_points(points2, idx, cuda=self.cuda, is_group=True) * weight.view(B, N, 3, 1), dim=2)
#             torch.cuda.empty_cache()
#
#         new_points = interpolated_points  ##########################
#
#         # if points1 is not None:
#         #     points1 = points1.permute(0, 2, 1)
#         #     new_points = torch.cat([points1, interpolated_points], dim=-1)
#         # else:
#         #     new_points = interpolated_points
#         #
#         # new_points = new_points.permute(0, 2, 1)
#         #
#         # # new_points = self.final(new_points.unsqueeze(-1)).squeeze(-1)   ############################
#         #
#         #
#         #
#         # for i, conv in enumerate(self.mlp_convs):
#         #     bn = self.mlp_bns[i]
#         #     # new_points = F.relu(bn(conv(new_points)))
#         #     new_points = bn(conv(new_points))  ###############################
#         return new_points

##############################################################################################

##############################################################################################

