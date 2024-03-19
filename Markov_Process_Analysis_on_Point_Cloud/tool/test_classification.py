import sys
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import time



sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(ROOT_DIR)
from dataset.ModelNetDataLoader import ModelNetDataLoader
from dataset.ScanObjectNNDataLoader import ScanObjectNNDataLoader
from util.utils import get_model, get_loss, set_seed, weight_init
from modules.pointnet2_utils import sample



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
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
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


class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            scales = torch.from_numpy(xyz).float().cuda()
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scales)
        return pc


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=40, help='batch size in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='res',  help='Experiment root')
    parser.add_argument('--model', default='repsurf.repsurf_ssg_umb', help='model file name [default: repsurf_ssg_umb]')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=10, help='Aggregate classification scores with voting')
    parser.add_argument('--cuda_ops', action='store_true', default=False,
                        help='Whether to use cuda version operations [default: False]')
    parser.add_argument('--n_workers', type=int, default=10, help='DataLoader Workers Number [default: 4]')
    parser.add_argument('--seed', type=int, default=2800, help='Training Seed')


    # Modeling
    parser.add_argument('--return_dist', action='store_true', default=True,
                        help='Whether to use signed distance [default: False]')
    parser.add_argument('--return_center', action='store_true', default=True,
                        help='Whether to return center in surface abstraction [default: False]')
    parser.add_argument('--return_polar', action='store_true', default=False,
                        help='Whether to return polar coordinate in surface abstraction [default: False]')
    parser.add_argument('--group_size', type=int, default=8, help='Size of umbrella group [default: 8]')
    parser.add_argument('--umb_pool', type=str, default='sum', help='pooling for umbrella repsurf [sum, mean, max]')

    return parser.parse_args()



def test(model, loader, num_class=15, vote_num=10):
    NUM_REPEAT = 50
    best_acc = 0
    best_ins = 0
    pointscale = PointcloudScale(scale_low=0.95, scale_high=1.05)
    for i in range(NUM_REPEAT):
        mean_correct = []

        class_acc = np.zeros((num_class,3))
        for j, data in tqdm(enumerate(loader), total=len(loader)):
            points, target = data

            points, target = points.cuda(), target.cuda()

            # preprocess
            points = sample(args.num_point, points, cuda=args.cuda_ops)
            points = points.transpose(2, 1)



            classifier = model.eval()
            vote_pool = torch.zeros(target.size()[0],num_class).cuda()
            for v in range(vote_num):
                if v > 0:
                    points.data = pointscale(points.data)

                points = points.permute(0,2,1)
                pred = classifier(points)
                points = points.permute(0,2,1)

                vote_pool += pred
            pred = vote_pool/vote_num
            pred_choice = pred.data.max(1)[1]
            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
                class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
                class_acc[cat,1]+=1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))
        class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
        class_acc = np.mean(class_acc[:,2])
        instance_acc = np.mean(mean_correct)
        print(instance_acc,class_acc)
        if instance_acc > best_ins:
            best_ins = instance_acc
            best_acc = class_acc


    return best_ins,best_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  ###############%%%%%%%%%%%%%%%%%
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #######################################


    '''CREATE DIR'''
    experiment_dir = '/home/XXX/Revisit/classification/log/ScanObjectNN/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = os.path.join('/Path/XXX/data', 'h5_files')

    TEST_DATASET = ScanObjectNNDataLoader(root=data_path, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.n_workers)


    args.num_class = 15

    '''MODEL LOADING'''

    classifier = get_model(args).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])


    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=args.num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
