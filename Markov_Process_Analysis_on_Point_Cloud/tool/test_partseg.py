import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

# sys.path.append(os.path.join(ROOT_DIR, 'models'))   #################%%%%%%%%%%%%%%%%%%%%%
sys.path.append(os.path.join(ROOT_DIR, 'log/part_seg/res'))    ##########################

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


# color_class = [[208,208,208],[151,203,255],[255,170,213],[173,254,220],[211,164,255],[211,255,147],[205,205,154],[255,237,151],[199,199,226],[255,189,157],
#                [60,60,60],[0,0,198],[100,166,0],[128,64,64],[123,123,123],[74,74,255],[154,255,2],[184,112,112],[96,0,0],[0,75,151],
#                [115,115,0],[128,128,64],[206,0,0],[0,128,255],[225,225,0],[175,175,97],[159,0,80],[0,121,121],[151,124,0],[64,128,128],
#                [255,0,128],[0,227,227],[234,193,0],[111,183,183],[117,0,117],[1,152,88],[187,94,0],[90,90,173],[232,0,232],[2,247,142],
#                [255,146,36],[153,153,204],[75,0,145],[0,145,0],[162,52,0],[143,69,134],[146,26,255],[0,236,0],[255,88,9],[183,102,173]]

color_class = [[254,30,30],[251,193,23],[169,169,169],[245,111,47],[42,42,42],[235,144,55]]

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, default='res',  help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


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


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  ###############%%%%%%%%%%%%%%%%%
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"  #######################################

    experiment_dir = 'log/part_seg/' + args.log_dir

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


    root = '/Path/XXX/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():

        #######################################################################################
        best_acc = 0
        best_cls_acc = 0
        best_cls_mIou = 0
        best_ins_mIou = 0
        NUM_REPEAT = 1
        pointscale = PointcloudScale(scale_low=0.95, scale_high=1.05)
        classifier = classifier.eval()

        for i in range(NUM_REPEAT):

            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

                if batch_id >= 0:
                    batchsize, num_point, _ = points.size()
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)
                    vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

                    for v in range(args.num_votes):
                        if v > 0:
                            points.data = pointscale(points.data)
                        seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                        vote_pool += seg_pred

                    seg_pred = vote_pool / args.num_votes
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1)  # + seg_classes[cat][0]



                    correct = np.sum(cur_pred_val == target)
                    total_correct += correct
                    total_seen += (cur_batch_size * NUM_POINT)

                    for l in range(num_part):
                        total_seen_class[l] += np.sum(target == l)
                        total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))




            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            # for cat in sorted(shape_ious.keys()):
            #     log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

            print(test_metrics['inctance_avg_iou'])
            if test_metrics['inctance_avg_iou'] > best_ins_mIou:
                best_acc = test_metrics['accuracy']
                best_cls_acc = test_metrics['class_avg_accuracy']
                best_cls_mIou = test_metrics['class_avg_iou']
                best_ins_mIou = test_metrics['inctance_avg_iou']
            print("best instance mIou: " % best_ins_mIou)

    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))

    log_string('Accuracy is: %.5f' % best_acc)
    log_string('Class avg accuracy is: %.5f' % best_cls_acc)
    log_string('Class avg mIOU is: %.5f' % best_cls_mIou)
    log_string('Inctance avg mIOU is: %.5f' % best_ins_mIou)



if __name__ == '__main__':
    args = parse_args()
    main(args)
