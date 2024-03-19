import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from collections import OrderedDict
from models.polar_utils import xyz2sphere
from models.recons_utils import cal_const, cal_normal, cal_center, check_nan_umb
# from models.pointnet2_utils_rep import query_knn_point, farthest_point_sample, index_points


def upsample(points, knn_idx, scale_ratio=2, dist=None):
    """
    Input:
        small points: input points data, [B, S, C]
        scale_ratio: upsampling ratio
        knn_idx: sample index data, [B, S, K]
    Return:
        big_points:, indexed points data, [B, N, C]
    """
    # knn_idx = knn_idx[:,:,:3]
    # dist = dist[:,:,:3]

    B, S, C = points.shape
    _, _, K = knn_idx.shape

    points1 = points.unsqueeze(-2).repeat(1, 1, K, 1)

    # dist_recip = 1.0 / (dist + 1e-8)
    # norm = torch.sum(dist_recip, dim=2, keepdim=True)
    # weight = dist_recip / norm

    # points1 = points1 #* weight.unsqueeze(-1)

    big = torch.cuda.FloatTensor(B, S, S * scale_ratio, C).zero_()
    # big = big.detach_()

    idx = knn_idx.unsqueeze(-1).repeat(1, 1, 1, C)
    out = big.scatter_(-2, idx.long(), points1)

    sum = torch.sum(out, dim=1)

    non_zero = torch.count_nonzero(out[:, :, :, 0], dim=1).unsqueeze(-1).float()
    one = torch.ones_like(non_zero)
    non_zero = torch.where(non_zero == 0.0, one, non_zero)

    out = sum / non_zero

    return out


def mod_index(bse_xyz, mod_idx, xyz):
    # bse_xyz in [B, N, D]
    batch_indexes = torch.arange(mod_idx.shape[0]).unsqueeze(dim=-1).repeat(1, mod_idx.shape[1]).flatten()
    modidx_indexes = mod_idx.flatten()
    mask = torch.ones_like(bse_xyz)
    mask[batch_indexes, modidx_indexes, :] = 0.0
    expand_xyz = torch.zeros_like(bse_xyz)
    expand_xyz[batch_indexes, modidx_indexes, :] = xyz.view(-1, xyz.shape[-1])
    return bse_xyz * mask + expand_xyz


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

def knn_point2(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """


    sqrdists = square_distance(new_xyz, xyz)

    B, N, _ = sqrdists.shape
    repeat_shape = [B, 1, 1]
    eye = torch.eye(N).unsqueeze(0) + 1.0
    eye = eye.repeat(repeat_shape).cuda()
    zeros = torch.zeros_like(sqrdists).cuda()
    zeros2 = torch.where(eye>1, zeros, eye)

    noise = torch.randn(zeros.size()).cuda() #* 1e-4
    big_num = zeros + 10 + noise
    sqrdists = torch.where(sqrdists==0, big_num, sqrdists)

    sqrdists = sqrdists * zeros2

    dist, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    return dist, group_idx

def random_sample(xyz, sample_num):
    B, N, _ = xyz.size()
    permutation = torch.randperm(N)
    temp_sample = xyz[:, permutation]
    sampled_xyz = temp_sample[:, :sample_num, :]

    idx = permutation[:sample_num].unsqueeze(0).expand(B, sample_num)

    return sampled_xyz, idx

def convert_polar(neighbours, center):
    neighbours = neighbours.permute(0, 2, 3, 1).contiguous()
    center = center.permute(0, 2, 3, 1).contiguous()

    rel_x = (neighbours - center)[:, :, :, 0]
    rel_y = (neighbours - center)[:, :, :, 1]
    rel_z = (neighbours - center)[:, :, :, 2]

    r_xy = torch.sqrt(rel_x ** 2 + rel_y ** 2)
    r_zx = torch.sqrt(rel_z ** 2 + rel_x ** 2)
    r_yz = torch.sqrt(rel_y ** 2 + rel_y ** 2)

    ### Z_axis
    z_beta = torch.atan2(rel_z, r_xy).unsqueeze(-3).contiguous()
    z_alpha = torch.atan2(rel_y, rel_x).unsqueeze(-3).contiguous()

    ### Y_axis
    y_beta = torch.atan2(rel_y, r_zx).unsqueeze(-3).contiguous()
    y_alpha = torch.atan2(rel_x, rel_z).unsqueeze(-3).contiguous()

    ### X_axis
    x_beta = torch.atan2(rel_x, r_yz).unsqueeze(-3).contiguous()
    x_alpha = torch.atan2(rel_z, rel_y).unsqueeze(-3).contiguous()

    return x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta

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
    # idx = query_knn_point(k, xyz, new_xyz, cuda=cuda)
    _, idx = knn_point(k, xyz, new_xyz)  ### B N K  #############################
    # torch.cuda.empty_cache()
    # group_xyz = index_points(xyz, idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
    group_xyz = index_points(xyz, idx)[:, :, 1:]  # [B, N', K-1, 3]  ####################
    # torch.cuda.empty_cache()

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
            # nn.Linear(in_channel, in_channel),
            # nn.LayerNorm(in_channel),
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.Linear(in_channel, in_channel),
            # nn.LayerNorm(in_channel),
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.Linear(in_channel, in_channel)
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
            # group_normal, group_center, group_polar, group_pos = check_nan_umb(group_normal, group_center, group_polar, group_pos)   ###############
            new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        else:
            group_normal, group_center = check_nan_umb(group_normal, group_center)
            new_feature = torch.cat([group_center, group_polar, group_normal], dim=-1)
        new_feature = new_feature.permute(0, 3, 2, 1)  # [B, C, G, N]

        # mapping
        new_feature = self.mlps(new_feature)
        # new_feature = new_feature.permute(0, 3, 2, 1)  # [B, C, G, N]

        # aggregation
        if self.aggr_type == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature

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
        # self.fc1 = Linear(out_channels*2, out_channels, bn=False)
        self.fc2 = Linear(out_channels*3, out_channels, bn=False)
        # self.conv_normal = nn.Linear(10, out_channels)
        # self.conv_normal2 = nn.Linear(10, out_channels)

        self.xyz_Trans = LocalTrans(3, out_channels, knn, usetanh=self.usetanh, residual=True)
        self.normal_Trans = LocalTrans(10, out_channels, knn, usetanh=self.usetanh, residual=True)
        self.feature_Trans1 = LocalTrans(in_channels, out_channels, knn, usetanh=self.usetanh, residual=self.residual)
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
            xyz_f = self.xyz_Trans(features=base_xyz, idx=idx, pos=base_xyz, FPS_idx=FPS_idx, xyz=True)
            # normal_f = self.normal_Trans(features=normal, idx=idx, pos=base_xyz, FPS_idx=FPS_idx)
            features1 = self.feature_Trans1(features=feature, idx=idx, pos=base_xyz, FPS_idx=FPS_idx)
            features2 = self.feature_Trans2(features=feature, idx=idx_feature, pos=base_xyz, FPS_idx=FPS_idx)

            merge_features = self.fc2(torch.cat((xyz_f, features1, features2), dim=2))

        if FPS_idx is not None:
            normal = index_points(normal, FPS_idx)
            # normal = normal[FPS_idx.long(), :]


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

        # self.pos_emb = Linear(3, in_c, bn=False)

        # self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def forward(self, features, idx, pos, FPS_idx=None, xyz=False):

        # pos_emb = self.pos_emb(pos)
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

class Fuse(nn.Module):
    def __init__(self, c0, c1, c2, c3, c4):
        super(Fuse, self).__init__()

        self.knn =8

        self.conv04 = Linear(c0, c4, bn=False)
        self.conv14 = Linear(c1, c4, bn=False)
        self.conv24 = Linear(c2, c4, bn=False)
        self.conv34 = Linear(c3, c4, bn=False)
        self.conv4 = Linear(c4, c4, bn=False)

        self.conv03 = Linear(c0, c3, bn=False)
        self.conv13 = Linear(c1, c3, bn=False)
        self.conv23 = Linear(c2, c3, bn=False)
        self.conv43 = Linear(c4, c3, bn=False)
        self.conv3 = Linear(c3, c3, bn=False)

        self.conv02 = Linear(c0, c2, bn=False)
        self.conv12 = Linear(c1, c2, bn=False)
        self.conv32 = Linear(c3, c2, bn=False)
        self.conv42 = Linear(c4, c2, bn=False)
        self.conv2 = Linear(c2, c2, bn=False)

        self.conv01 = Linear(c0, c1, bn=False)
        self.conv21 = Linear(c2, c1, bn=False)
        self.conv31 = Linear(c3, c1, bn=False)
        self.conv41 = Linear(c4, c1, bn=False)
        self.conv1 = Linear(c1, c1, bn=False)

        self.conv10 = Linear(c1, c0, bn=False)
        self.conv20 = Linear(c2, c0, bn=False)
        self.conv30 = Linear(c3, c0, bn=False)
        self.conv40 = Linear(c4, c0, bn=False)
        self.conv0 = Linear(c0, c0, bn=False)

    def forward(self, num_point, f0=None, f1=None, f2=None, f3=None, f4=None, FPS_0=None, FPS_1=None, FPS_2=None, FPS_3=None,
                knn_0=None, knn_1=None, knn_2=None, knn_3=None, knn_4=None, xyz0=None, xyz1=None, xyz2=None, xyz3=None, xyz4=None):

        if num_point == 128:
            ###04
            idx04 = index_points(FPS_0.unsqueeze(-1),
                     index_points(FPS_1.unsqueeze(-1),
                       index_points(FPS_2.unsqueeze(-1), FPS_3).squeeze(-1)).squeeze(-1)).squeeze(-1)
            f04 = self.conv04(index_points(f0, idx04))

            ###14
            idx14 = index_points(FPS_1.unsqueeze(-1),
                       index_points(FPS_2.unsqueeze(-1), FPS_3).squeeze(-1)).squeeze(-1)
            f14 = self.conv14(index_points(f1, idx14))

            ###24
            idx24 = index_points(FPS_2.unsqueeze(-1), FPS_3).squeeze(-1)
            f24 = self.conv24(index_points(f2, idx24))

            ###34
            f34 = self.conv34(index_points(f3, FPS_3))

            f4 = self.conv4(f4 + f04 + f14 + f24 + f34) + f4

        if num_point == 256:
            ###03
            idx03 = index_points(FPS_0.unsqueeze(-1),
                     index_points(FPS_1.unsqueeze(-1), FPS_2).squeeze(-1)).squeeze(-1)
            f03 = self.conv03(index_points(f0, idx03))

            ###13
            idx13 = index_points(FPS_1.unsqueeze(-1), FPS_2).squeeze(-1)
            f13 = self.conv13(index_points(f1, idx13))

            ###23
            f23 = self.conv23(index_points(f2, FPS_2))

            ###43
            f43 = self.conv43(upsample(f4, knn_4))

            f3 = self.conv3(f3 + f03 + f13 + f23 + f43) + f3

        if num_point == 512:
            ###02
            idx02 = index_points(FPS_0.unsqueeze(-1), FPS_1).squeeze(-1)
            f02 = self.conv02(index_points(f0, idx02))

            ###12
            f12 = self.conv12(index_points(f1, FPS_1))

            ###32
            f32 = self.conv32(upsample(f3, knn_3))

            ###42

            _, knn_42 = knn_point(self.knn, xyz2, xyz4)
            f42 = self.conv42(upsample(f4, knn_42, scale_ratio=4))   #############

            f2 = self.conv2(f2 + f02 + f12 + f32 + f42) + f2


        if num_point == 1024:
            ###01
            f01 = self.conv01(index_points(f0, FPS_0))

            ###21
            f21 = self.conv21(upsample(f2, knn_2))

            ###31
            _, knn_31 = knn_point(self.knn, xyz1, xyz3)
            f31 = self.conv31(upsample(f3, knn_31, scale_ratio=4))

            ###41
            _, knn_41 = knn_point(self.knn, xyz1, xyz4)
            f41 = self.conv41(upsample(f4, knn_41, scale_ratio=8))

            f1 = self.conv1(f1 + f01 + f21 + f31 + f41) + f1


        if num_point == 2048:
            ###10
            f10 = self.conv10(upsample(f1, knn_1))

            ###20
            _, knn_20 = knn_point(self.knn, xyz0, xyz2)
            f20 = self.conv20(upsample(f2, knn_20, scale_ratio=4))

            ###30
            _, knn_30 = knn_point(self.knn, xyz0, xyz3)
            f30 = self.conv30(upsample(f3, knn_30, scale_ratio=8))

            ###40
            _, knn_40 = knn_point(self.knn, xyz0, xyz4)
            f40 = self.conv40(upsample(f4, knn_40, scale_ratio=16))

            f0 = self.conv0(f0 + f10 + f20 + f30 + f40) + f0

        return f0, f1, f2, f3, f4

class KeepHighResolutionModulePartSeg(nn.Module):

    def __init__(self, data_C, b1_C, b2_C, b3_C, b4_C, cuda=False):
        super(KeepHighResolutionModulePartSeg, self).__init__()


        self.neighbour = 16
        self.cuda = cuda

        self.start = Linear(3, 32, bn=False)

        self.la0 = LocalMerge(32, 64, 8, usetanh=False, residual=True)
        self.la1 = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        self.la2 = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        self.la3 = LocalMerge(64, 128, 8, usetanh=False, residual=True)
        self.la4 = LocalMerge(128, 256, 8, usetanh=False, residual=True)


        # self.la0_1 = LocalMerge(64, 64, 8, usetanh=False, residual=True)
        # self.la1_1 = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        # self.la2_1 = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        # self.la3_1 = LocalMerge(64, 128, 8, usetanh=False, residual=True)
        # self.la4_1 = LocalMerge(128, 256, 8, usetanh=False, residual=True)



        # self.la1_g = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        # self.la2_g = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        # self.la3_g = LocalMerge(128, 128, 8, usetanh=False, residual=False)


        # self.la5_up = LocalMerge(256, 256, 8, usetanh=False, residual=False)
        self.la4_up = LocalMerge(128, 128, 8, usetanh=False, residual=False)
        self.la3_up = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        self.la2_up = LocalMerge(64, 64, 8, usetanh=False, residual=False)
        self.la1_up = LocalMerge(64, 64, 8, usetanh=False, residual=False)


        # self.up_conv5 = Linear(512, 256, bn=False)
        self.up_conv4 = Linear(256, 128, bn=False)
        self.up_conv3 = Linear(128, 64, bn=False)
        self.up_conv2 = Linear(64, 64, bn=False)
        self.up_conv1 = Linear(64, 64, bn=False)

        self.mlp = Linear(256, 256, bn=False)
        # self.mlp_la = LocalMerge(256, 256, 8, usetanh=False, residual=False)

        # self.conv1 = Linear(128, 64, bn=False)
        # self.conv2 = Lnear(128, 64, bn=False)
        # self.conv3 = Linear(128, 64, bn=False)
        # self.conv4 = Linear(256, 128, bn=False)

        self.conv5 = Linear(64, 256, bn=False)
        self.conv6 = Linear(64, 128, bn=False)
        self.conv7 = Linear(16, 64, bn=False)
        self.conv8 = Linear(64, 256, bn=False)

        self.fuse1 = Fuse(64, 64, 64, 128, 256)
        self.fuse2 = Fuse(64, 64, 64, 128, 256)
        self.fuse3 = Fuse(64, 64, 64, 128, 256)
        self.fuse4 = Fuse(64, 64, 64, 128, 256)
        self.fuse5 = Fuse(64, 64, 64, 128, 256)


        # self.final_class = nn.Linear(1024, 1024)
        # self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, xyz, normal, label):

        xyz = xyz.permute(0, 2, 1).contiguous()
        normal = normal.permute(0, 2, 1).contiguous()

        _, num_points, _ = xyz.shape

        # start = self.start(xyz)
        # points, normal0, knn_idx0, dist0 = self.la0(xyz=xyz, base_xyz=xyz, normal=normal, feature=start, xyz_flag=True)

        points, normal0, knn_idx0, dist0 = self.la0(xyz=xyz, base_xyz=xyz, normal=normal, xyz_flag=True)


        # # points_temp = points.detach()
        # _, FPS_idx0, normal, new_offset = fps(self.stride, self.num_sector, xyz, normal, offset, self.training)
        # FPS_xyz0 = xyz[FPS_idx0.long(), :]
        FPS_idx0 = farthest_point_sample(xyz, 1024)
        # torch.cuda.empty_cache()
        FPS_xyz0 = index_points(xyz, FPS_idx0)

        # FPS_idx0 = farthest_point_sample(points, 512, cuda=self.cuda)
        # torch.cuda.empty_cache()
        # FPS_xyz0 = index_points(xyz, FPS_idx0, cuda=self.cuda, is_group=False)
        # torch.cuda.empty_cache()

        points0, normal1, knn_idx1, dist1 = self.la1(xyz=FPS_xyz0, base_xyz=xyz, normal=normal0, feature=points, FPS_idx=FPS_idx0, xyz_flag=True)
        # points0, _, _, _ = self.la1_g(xyz=FPS_xyz0, base_xyz=FPS_xyz0, normal=normal, feature=points0)
        # torch.cuda.empty_cache()



        # _, FPS_idx1, normal, new_offset1 = fps(self.stride, 0, FPS_xyz0, normal, new_offset, self.training)
        # FPS_xyz1 = FPS_xyz0[FPS_idx1.long(), :]
        FPS_idx1 = farthest_point_sample(FPS_xyz0, 512)
        # torch.cuda.empty_cache()
        FPS_xyz1 = index_points(FPS_xyz0, FPS_idx1)

        # FPS_idx1 = farthest_point_sample(points0, 256, cuda=self.cuda)
        # torch.cuda.empty_cache()
        # FPS_xyz1 = index_points(FPS_xyz0, FPS_idx1, cuda=self.cuda, is_group=False)
        # torch.cuda.empty_cache()

        point1, normal2, knn_idx2, dist2 = self.la2(xyz=FPS_xyz1, base_xyz=FPS_xyz0, normal=normal1, feature=points0, FPS_idx=FPS_idx1, xyz_flag=False)
        # point1, _, _, _ = self.la2_g(xyz=FPS_xyz1, base_xyz=FPS_xyz1, normal=normal, feature=point1)
        # torch.cuda.empty_cache()



        # _, idx2, normal, new_offset2 = fps(self.stride, 0, FPS_xyz1, normal, new_offset1, self.training)
        # branch2_xyz = FPS_xyz1[idx2.long(), :]
        idx2 = farthest_point_sample(FPS_xyz1, 256)
        # torch.cuda.empty_cache()
        branch2_xyz = index_points(FPS_xyz1, idx2)

        # idx2 = farthest_point_sample(point1, 128, cuda=self.cuda)
        # torch.cuda.empty_cache()
        # branch2_xyz = index_points(branch1_xyz, idx2, cuda=self.cuda, is_group=False)
        # torch.cuda.empty_cache()

        branch2_points_FP, normal3, knn_idx3, dist3 = self.la3(xyz=branch2_xyz, base_xyz=FPS_xyz1, normal=normal2,
                                                       feature=point1, FPS_idx=idx2, xyz_flag=True)
        # branch2_points_FP, _, _, _ = self.la3_g(xyz=branch2_xyz, base_xyz=branch2_xyz, normal=normal,
        #                                                feature=branch2_points_FP)
        # torch.cuda.empty_cache()


        # branch2_points_FP_temp = branch2_points_FP.detach()
        # _, idx3, normal, new_offset3 = fps(self.stride, 0, branch2_xyz, normal, new_offset2, self.training)
        # branch3_xyz = branch2_xyz[idx3.long(), :]
        idx3 = farthest_point_sample(branch2_xyz, 128)
        # torch.cuda.empty_cache()
        branch3_xyz = index_points(branch2_xyz, idx3)

        # idx3 = farthest_point_sample(branch2_points_FP, 64, cuda=self.cuda)
        # torch.cuda.empty_cache()
        # branch3_xyz = index_points(branch2_xyz, idx3, cuda=self.cuda, is_group=False)
        # torch.cuda.empty_cache()

        branch3_points_FP, normal4, knn_idx4, dist4 = self.la4(xyz=branch3_xyz, base_xyz=branch2_xyz, normal=normal3,
                                                       feature=branch2_points_FP, FPS_idx=idx3, xyz_flag=False)




        # torch.cuda.empty_cache()

        # # branch3_points_FP_temp = branch3_points_FP.detach()
        # # _, idx4, normal, new_offset4 = fps(self.stride, 0, branch3_xyz, normal, new_offset3, self.training)
        # # branch4_xyz = branch3_xyz[idx4.long(), :]
        # idx4 = farthest_point_sample(branch3_points_FP, 64, cuda=True)
        # # torch.cuda.empty_cache()
        # branch4_xyz = index_points(branch3_xyz, idx4, cuda=True, is_group=False)
        #
        # # idx4 = farthest_point_sample(branch3_points_FP, 32, cuda=self.cuda)
        # # torch.cuda.empty_cache()
        # # branch4_xyz = index_points(branch3_xyz, idx4, cuda=self.cuda, is_group=False)
        # # torch.cuda.empty_cache()
        #
        # branch4_points_FP, normal, knn_idx5 = self.la5(xyz=branch4_xyz, base_xyz=branch3_xyz, normal=normal,
        #                                                feature=branch3_points_FP, FPS_idx=idx4)
        # # torch.cuda.empty_cache()
        #
        # ##upsample
        # #############################
        #
        # # branch3_points_FP = branch3_points_FP + self.up_fp4(branch3_xyz, branch4_xyz, branch3_points_FP, branch4_points_FP)
        # branch3_points_FP_up, _, _ = self.la5_up(xyz=branch3_xyz, base_xyz=branch3_xyz,
        #                                          feature=self.up_conv5(upsample(branch4_points_FP, knn_idx5.long())))
        # # torch.cuda.empty_cache()
        # branch3_points_FP = branch3_points_FP_up + branch3_points_FP
        #
        # # branch3_points_FP_up = self.conv5(up(branch4_points_FP, new_offset4, new_offset3).unsqueeze(1)).squeeze(1)
        # # branch3_points_FP_up_temp, normal = self.la5_up(xyz=branch3_xyz, base_xyz=branch3_xyz, normal=normal, offset=new_offset3,
        # #                            feature=branch3_points_FP)
        # # # branch3_points_FP_up_temp = replace(branch3_points_FP_up_temp, new_offset4, new_offset3)
        # # branch3_points_FP_up = branch3_points_FP_up_temp + branch3_points_FP_up
        #
        # # branch3_points_FP_up = self.la5_up(branch3_points_FP_up, branch3_points_FP, branch3_xyz, new_offset3)
        #
        # # branch2_points_FP = branch2_points_FP + self.up_fp3(branch2_xyz, branch3_xyz, branch2_points_FP, branch3_points_FP)

        branch3_points_FP_up = self.mlp(branch3_points_FP)


        _, _, _, _, branch3_points_FP_up = self.fuse1(128, f0=points, f1=points0, f2=point1, f3=branch2_points_FP, f4=branch3_points_FP_up,
                                                   FPS_0=FPS_idx0, FPS_1=FPS_idx1, FPS_2=idx2, FPS_3=idx3)

        # branch3_points_FP_up += branch3_points_FP
        # branch3_points_FP, _, _, _ = self.mlp_la(xyz=branch3_xyz, base_xyz=branch3_xyz, normal=normal, feature=branch3_points_FP)

        # branch2_points_FP_up = self.up_conv4(upsample(branch3_points_FP_up, knn_idx4, dist=dist4))
        branch2_points_FP_up, _, _, _ = self.la4_up(xyz=branch2_xyz, base_xyz=branch2_xyz, normal=normal3,
                                                 feature=self.up_conv4(upsample(branch3_points_FP_up, knn_idx4, dist=dist4)))


        _, _, _, branch2_points_FP_up, _ = self.fuse2(256, f0=points, f1=points0, f2=point1, f3=branch2_points_FP_up, f4=branch3_points_FP,
                                                   FPS_0=FPS_idx0, FPS_1=FPS_idx1, FPS_2=idx2, FPS_3=idx3,
                                                      knn_0=knn_idx0, knn_1=knn_idx1, knn_2=knn_idx2, knn_3=knn_idx3, knn_4=knn_idx4)
        # branch2_points_FP_up += branch2_points_FP
        # torch.cuda.empty_cache()
        # branch2_points_FP = self.conv4(torch.cat((branch2_points_FP, branch2_points_FP_up), dim=-1))# branch2_points_FP_up + branch2_points_FP

        # point1_up = self.up_conv3(upsample(branch2_points_FP_up, knn_idx3, dist=dist3))
        point1_up, _, _, _ = self.la3_up(xyz=FPS_xyz1, base_xyz=FPS_xyz1, normal=normal2,
                                      feature=self.up_conv3(upsample(branch2_points_FP_up, knn_idx3, dist=dist3)))

        # point1_up = self.up_conv3(upsample(branch2_points_FP, knn_idx3.long(), dist3))
        # torch.cuda.empty_cache()
        # point1 = self.conv3(torch.cat((point1, point1_up), dim=-1))#point1_up + point1
        # point1_up = self.conv3(up(branch2_points_FP_up, new_offset2, new_offset1).unsqueeze(1)).squeeze(1)
        # point1_up_temp, normal = self.la3_up(xyz=FPS_xyz1, base_xyz=FPS_xyz1, normal=normal, offset=new_offset1,
        #                            feature=point1)
        # # point1_up_temp = replace(point1_up_temp, new_offset2, new_offset1)
        # point1_up = point1_up_temp + point1_up
        # point1_up = self.la3_up(point1_up, point1, FPS_xyz1, new_offset1)

        # points0 = points0 + self.up_fp1(FPS_xyz0, FPS_xyz1, points0, point1)

        _, _, point1_up, _, _ = self.fuse3(512, f0=points, f1=points0, f2=point1_up, f3=branch2_points_FP, f4=branch3_points_FP,
                                                   FPS_0=FPS_idx0, FPS_1=FPS_idx1, FPS_2=idx2, FPS_3=idx3,
                                                      knn_0=knn_idx0, knn_1=knn_idx1, knn_2=knn_idx2, knn_3=knn_idx3, knn_4=knn_idx4,
                                        xyz0=xyz, xyz1=FPS_xyz0, xyz2=FPS_xyz1, xyz3=branch2_xyz, xyz4=branch3_xyz)

        # point1_up += point1
        # points0_up = self.up_conv2(upsample(point1_up, knn_idx2, dist=dist2))
        points0_up, _, _, _ = self.la2_up(xyz=FPS_xyz0, base_xyz=FPS_xyz0, normal=normal1,
                                      feature=self.up_conv2(upsample(point1_up, knn_idx2, dist=dist2)))

        # points0_up = self.up_conv2(upsample(point1, knn_idx2.long(), dist2))
        # torch.cuda.empty_cache()
        # points0 = self.conv2(torch.cat((points0, points0_up), dim=-1))#points0_up + points0

        _, points0_up, _, _, _ = self.fuse4(1024, f0=points, f1=points0_up, f2=point1, f3=branch2_points_FP, f4=branch3_points_FP,
                                                   FPS_0=FPS_idx0, FPS_1=FPS_idx1, FPS_2=idx2, FPS_3=idx3,
                                                      knn_0=knn_idx0, knn_1=knn_idx1, knn_2=knn_idx2, knn_3=knn_idx3, knn_4=knn_idx4,
                                        xyz0=xyz, xyz1=FPS_xyz0, xyz2=FPS_xyz1, xyz3=branch2_xyz, xyz4=branch3_xyz)

        # points0_up += points0
        # points_up = self.up_conv1(upsample(points0_up, knn_idx1, dist=dist1))
        points_up, _, _, _ = self.la1_up(xyz=xyz, base_xyz=xyz, normal=normal0,
                                      feature=self.up_conv1(upsample(points0_up, knn_idx1, dist=dist1)))


        # points_up = self.up_conv1(upsample(points0, knn_idx1.long(), dist1))
        # torch.cuda.empty_cache()
        # points = self.conv1(torch.cat((points, points_up), dim=-1))#points_up + points

        points_up, _, _, _, _ = self.fuse5(2048, f0=points_up, f1=points0, f2=point1, f3=branch2_points_FP, f4=branch3_points_FP,
                                                   FPS_0=FPS_idx0, FPS_1=FPS_idx1, FPS_2=idx2, FPS_3=idx3,
                                                      knn_0=knn_idx0, knn_1=knn_idx1, knn_2=knn_idx2, knn_3=knn_idx3, knn_4=knn_idx4,
                                        xyz0=xyz, xyz1=FPS_xyz0, xyz2=FPS_xyz1, xyz3=branch2_xyz, xyz4=branch3_xyz)

        # points_up += points
        global_rep = torch.cat((F.adaptive_max_pool1d(points_up.permute(0,2,1), 1), F.adaptive_max_pool1d(points0_up.permute(0,2,1), 1),
                       F.adaptive_max_pool1d(point1_up.permute(0,2,1), 1), F.adaptive_max_pool1d(branch2_points_FP_up.permute(0,2,1), 1),
                       F.adaptive_max_pool1d(branch3_points_FP_up.permute(0,2,1), 1)), dim=1)

        global_rep = global_rep.permute(0,2,1).repeat(1, num_points, 1)

        label = self.conv7(label)
        label = label.repeat(1, num_points, 1)

        points_up = self.conv5(points_up)
        final = torch.cat((points_up, global_rep, label), 2)


        ######################################################################
        # x = self.conv6(points_up)
        # x = x.max(dim=1, keepdim=True)[0]
        #
        # label = self.conv7(label)
        #
        # points_up = self.conv8(points_up)
        #
        # x = torch.cat((x, label), dim=2)
        # x = x.repeat(1, num_points, 1)
        #
        # final = torch.cat((x, points_up), 2)

        return xyz, final#, ret2, ret3, ret4

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, act=False):
        super(PointNetFeaturePropagation, self).__init__()
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
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

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

