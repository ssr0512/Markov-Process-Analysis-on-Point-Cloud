import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import KeepHighResolutionModulePartSeg, UmbrellaSurfaceConstructor


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

class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        # self.cuda = True

        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        # self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        # self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        # self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        # self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

        ######################################################################

        # self.return_center = True
        # self.return_polar = False
        # self.init_nsample = 2048

        self.umb_pool = 'sum'
        self.group_size = 8
        self.return_dist = True
        # center_channel = 0 if not self.return_center else (6 if self.return_polar else 3)
        repsurf_channel = 10

        self.surface_constructor = UmbrellaSurfaceConstructor(self.group_size + 1, repsurf_channel,
                                                              return_dist=self.return_dist, aggr_type=self.umb_pool,
                                                              cuda=False)

        self.keepHigh = KeepHighResolutionModulePartSeg(3, 64, 128, 256, 512, cuda=True)  ########################

        self.conv8 = Linear(896, 512, bn=False)
        self.conv9 = Linear(512, 256, bn=False)
        self.conv10 = Linear(256, 128, bn=False)
        self.conv11 = nn.Linear(128, num_classes)

        self.drop1 = nn.Dropout(0.5)  ########################
        self.drop2 = nn.Dropout(0.5)  ########################

        # self.bn8 = nn.BatchNorm2d(512)
        # self.bn9 = nn.BatchNorm2d(256)
        # self.bn10 = nn.BatchNorm2d(128)

        # self.conv8 = nn.Sequential(nn.Conv2d(1600, 512, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        # self.conv9 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        # self.conv10 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False), self.bn10, nn.LeakyReLU(negative_slope=0.2))
        # self.conv11 = nn.Conv2d(128, num_classes, kernel_size=1, bias=False)

        # self.conv1_loss = nn.Conv2d(128, num_classes, kernel_size=1, bias=False)
        # self.conv2_loss = nn.Conv2d(128, num_classes, kernel_size=1, bias=False)
        # self.conv3_loss = nn.Conv2d(128, num_classes, kernel_size=1, bias=False)
        # self.conv4_loss = nn.Conv2d(128, num_classes, kernel_size=1, bias=False)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        # B,C,N = xyz.shape
        # if self.normal_channel:
        #     l0_points = xyz
        #     l0_xyz = xyz[:,:3,:]
        # else:
        #     l0_points = xyz
        #     l0_xyz = xyz



        # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # # Feature Propagation layers
        # l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        # l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # # FC layers
        # feat = F.relu(self.bn1(self.conv1(l0_points)))
        # x = self.drop1(feat)
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        # return x, l3_points


        ######################################################



        # branch1_xyz, final_points, ret2, ret3, ret4 = self.keepHigh(xyz, label=cls_label)  ########################

        # center = xyz[:, :3, :]
        # noise = torch.randn(center.size()).cuda() * 1e-4
        # normal = self.surface_constructor(center + noise)
        # normal = center

        branch1_xyz, final_points = self.keepHigh(xyz, normal=xyz, label=cls_label)  ########################

        x = self.drop1(self.conv8(final_points))
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)

        # x = self.drop1(self.conv8(final_points))   ########################
        # x = self.drop2(self.conv9(x))    ########################
        # x = self.conv10(x)                     ########################
        # x = self.conv11(x).squeeze(-1)  ########################

        # x = F.log_softmax(x, dim=1)         ########################
        # x = x.permute(0, 2, 1)

        # x1 = self.conv1_loss(x1).squeeze(-1).permute(0, 2, 1)
        # x2 = self.conv2_loss(x2).squeeze(-1).permute(0, 2, 1)
        # x3 = self.conv3_loss(x3).squeeze(-1).permute(0, 2, 1)
        # x4 = self.conv4_loss(x4).squeeze(-1).permute(0, 2, 1)

        # return x, xyz, ret2, ret3, ret4
        return x, xyz


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # total_loss = F.nll_loss(pred, target)

        ########################################################################
        target = target.contiguous().view(-1)

        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        total_loss = -(one_hot * log_prb).sum(dim=1).mean()
        ###########################################################################
        # print('loss: ', total_loss)

        return total_loss

class get_loss2(nn.Module):
    def __init__(self):
        super(get_loss2, self).__init__()

        self.mi_loss = nn.BCEWithLogitsLoss()
        # self.mi_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, ret2, ret3, ret4):

        # print("ret2 :", ret2)
        # print("ret3 :", ret3)
        # t_s0 = torch.ones(ret0.shape[0], ret0.shape[1]//2)
        # f_s0 = torch.zeros(ret0.shape[0], ret0.shape[1]//2)
        #
        # t_s1 = torch.ones(ret1.shape[0], ret1.shape[1]//2)
        # f_s1 = torch.zeros(ret1.shape[0], ret1.shape[1]//2)

        t_s2 = torch.ones(ret2.shape[0], ret2.shape[1]//2)
        f_s2 = torch.zeros(ret2.shape[0], ret2.shape[1]//2)

        t_s3 = torch.ones(ret3.shape[0], ret3.shape[1]//2)
        f_s3 = torch.zeros(ret3.shape[0], ret3.shape[1]//2)

        t_s4 = torch.ones(ret4.shape[0], ret4.shape[1]//2)
        f_s4 = torch.zeros(ret4.shape[0], ret4.shape[1]//2)

        # s0 = torch.cat((t_s0, f_s0), 1).cuda()
        # s1 = torch.cat((t_s1, f_s1), 1).cuda()
        s2 = torch.cat((t_s2, f_s2), 1).cuda()
        s3 = torch.cat((t_s3, f_s3), 1).cuda()
        s4 = torch.cat((t_s4, f_s4), 1).cuda()

        # s2 = self.sigmoid(s2)
        # s3 = self.sigmoid(s3)
        # s4 = self.sigmoid(s4)

        # miloss_s0 = self.mi_loss(ret0, s0)
        # miloss_s1 = self.mi_loss(ret1, s1)
        miloss_s2 = self.mi_loss(ret2, s2)
        miloss_s3 = self.mi_loss(ret3, s3)
        miloss_s4 = self.mi_loss(ret4, s4)

        total_loss = (miloss_s2 + miloss_s3 + miloss_s4)/3

        return total_loss

