import torch.nn as nn
import torch.nn.functional as F
from modules.repsurface_utils import KeepHighResolutionModule
import torch

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

        self.init_nsample = args.num_point
        self.return_dist = args.return_dist

        self.keepHigh = KeepHighResolutionModule(3, 64, 64, 64, 64, cuda=args.cuda_ops)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, args.num_class)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)




    def forward(self, points):
        # init
        center = points[:, :3, :]

        normal = center

        final_points = self.keepHigh(center, normal)
        x = self.drop1(self.lrelu(self.bn1(self.fc1(final_points))))
        x = self.drop2(self.lrelu(self.bn2(self.fc2(x))))
        feature = self.fc3(x)


        feature = F.log_softmax(feature, -1)


        return feature
























