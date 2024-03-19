"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch.nn as nn
import torch.nn.functional as F
from modules.repsurface_utils import SurfaceAbstractionCD, UmbrellaSurfaceConstructor, KeepHighResolutionModule  ############
import torch   ######################

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

def inner_correlation(z, index=None):
    if index is not None:
        z = index_points(z, index)

    norm_z = F.normalize(z, dim=-1, p=2)
    corr_mat = torch.matmul(norm_z, norm_z.permute(0,2,1))

    return corr_mat


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        center_channel = 0 if not args.return_center else (6 if args.return_polar else 3)
        repsurf_channel = 10

        self.init_nsample = args.num_point
        self.return_dist = args.return_dist
        self.surface_constructor = UmbrellaSurfaceConstructor(args.group_size + 1, repsurf_channel,
                                                              return_dist=args.return_dist, aggr_type=args.umb_pool,
                                                              cuda=args.cuda_ops)

        self.sa1 = SurfaceAbstractionCD(npoint=512, radius=0.2, nsample=32, feat_channel=repsurf_channel,
                                        pos_channel=center_channel, mlp=[3, 64, 128], group_all=False,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        self.sa2 = SurfaceAbstractionCD(npoint=128, radius=0.4, nsample=64, feat_channel=128 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[131, 128, 256], group_all=False,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        self.sa3 = SurfaceAbstractionCD(npoint=None, radius=None, nsample=None, feat_channel=256 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[259, 512, 1024], group_all=True,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        # modelnet40
        self.classfier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(256, args.num_class))



        ####################################################################
        self.keepHigh = KeepHighResolutionModule(3, 64, 64, 64, 64, cuda=args.cuda_ops)  ########################
        self.fc1 = nn.Linear(1024, 512)     ########################
        self.bn1 = nn.BatchNorm1d(512)     ########################
        self.drop1 = nn.Dropout(0.5)     ########################
        self.fc2 = nn.Linear(512, 256)     ########################
        self.bn2 = nn.BatchNorm1d(256)     ########################
        self.drop2 = nn.Dropout(0.5)     ########################
        self.fc3 = nn.Linear(256, args.num_class)    ########################
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)    ########################
        ###########################################################################



    def forward(self, points):
        # init
        center = points[:, :3, :]

        normal = self.surface_constructor(center)

        ###############################################################
        # center, normal, feature = self.sa1(center, normal, None)
        # center, normal, feature = self.sa2(center, normal, feature)
        # center, normal, feature = self.sa3(center, normal, feature)
        #
        # feature = feature.view(-1, 1024)
        # feature = self.classfier(feature)
        ###############################################################



        # final_points, ret2, ret3, ret4 = self.keepHigh(center, normal)  ########################
        final_points = self.keepHigh(center, normal)  ########################
        # branch1_xyz, final_points = self.keepHigh(center, normal)  ########################
        # branch1_xyz, final_points, d1, d2, d3, d4, d5, d1_idx, d2_idx, d3_idx, d4_idx, d5_idx = self.keepHigh(xyz, norm)  ########################
        # final_xyz, final_points = self.finalGroup(branch1_xyz, branch1_points)    ########################
        # x = final_points.view(B, 256)                      ########################
        x = self.drop1(self.lrelu(self.bn1(self.fc1(final_points))))   ########################
        # x = self.lrelu(self.bn1(self.fc1(final_points)))   ########################
        x = self.drop2(self.lrelu(self.bn2(self.fc2(x))))    ########################
        feature = self.fc3(x)                     ########################



        feature = F.log_softmax(feature, -1)


        return feature
        # return feature, ret2, ret3, ret4























