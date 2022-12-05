from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart
from model import DGCNN_partseg, DGCNN_semantic_grasp
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement

from utils.points import regularize_pc_point_count

import pickle

def get_real_pc(cat, trial):
    with open(f'semantic/{cat}/trial{trial}/{cat}.pkl', 'rb') as fp:
        data = np.array(pickle.load(fp))
        pc = data[:,:3]
    pc = regularize_pc_point_count(pc, 2048, False)
    return pc


def read_pc(cat, id, view_size=4):


    with open(f'../grasp_network/data/pcs/{cat}/{cat}{id:03}.pkl', 'rb') as fp:
        data = pickle.load(fp)
        obj_pose_relative = data['obj_pose_relative']
        pcs = data['pcs']
    
    pc_indcs = np.random.randint(low=0, high=999, size=view_size)
    
    if len(pc_indcs) == 1:
        pc = pcs[pc_indcs[0]]
    else:
        __pcs = []
        for pc_index in pc_indcs:
            __pcs = __pcs + [pcs[pc_index]]
        pc = np.concatenate( __pcs )
   
    pc = regularize_pc_point_count(pc, 2048, False)
    return pc, obj_pose_relative


# def test(args):
#     device = torch.device("cuda" if args.cuda else "cpu")
    
#     #Try to load models
#     if args.model == 'dgcnn':
#         model = DGCNN_partseg(args, args.seg_num).to(device)
#     else:
#         raise Exception("Not implemented")

#     model = nn.DataParallel(model)
#     model.load_state_dict(torch.load(args.model_path))
#     model = model.eval()
#     test_acc = 0.0
#     count = 0.0
#     test_true_cls = []
#     test_pred_cls = []
#     test_true_seg = []
#     test_pred_seg = []
#     test_label_seg = []

#     #########################
#     # Get data from PC
#     #########################
#     # pc, _ = read_pc(cat='hammer', id=1, view_size=4)
    
#     pc = get_real_pc(cat=args.cat, trial=args.trial)
#     # pc = get_real_pc(cat='hammer', trial=3)
#     # pc -= np.mean(pc, axis=0)
#     # pc[:, 1] -=0.15

#     # print(pc.shape)
#     pc = np.expand_dims(pc, axis=0)
#     # print(pc.shape)
#     data = torch.from_numpy(pc.astype(np.float32))

#     # print(data.size())

#     seg = torch.zeros((1,2048)) + 7
#     # print(f"seg: {seg}")
#     # exit()

#     # seg = seg - seg_start_index

#     label = torch.Tensor([[7]]) # 7 - knife, 11 - mug
#     # label = torch.Tensor([[11]]) # 7 - knife, 11 - mug
#     label_one_hot = np.zeros((1, 16))
#     label_one_hot[0,7] = 1
#     label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))

#     # label_one_hot = np.zeros((label.shape[0], 16))
#     # for idx in range(label.shape[0]):
#     #     label_one_hot[idx, label[idx]] = 1
#     # label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))



#     data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
#     data = data.permute(0, 2, 1)
#     batch_size = data.size()[0]
#     print(data.size())
#     print(label_one_hot.size())
#     seg_pred = model(data, label_one_hot)
#     print(seg_pred)
#     print(seg_pred.shape)
    
#     # exit()
#     seg_pred = seg_pred.permute(0, 2, 1).contiguous()
#     pred = seg_pred.max(dim=2)[1]
#     # visiualization
#     np.savez_compressed('/home/gpupc2/GRASP/grasper/pat_semantic/hammer', data=data.cpu(), pred=pred.cpu())

def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN_semantic_grasp(args).to(device)

    quat = torch.Tensor([0,0,0,1]).to(device)    
    trans = torch.Tensor([0,0,0]).to(device)    

    pc, _ = read_pc(cat='hammer', id=1, view_size=4)
    pc = np.expand_dims(pc, axis=0)
    pc = torch.from_numpy(pc.astype(np.float32)).to(device)
    pc = pc.permute(0, 2, 1).contiguous()
    

    label_one_hot = torch.zeros((1, 16)).to(device)
    label_one_hot[0,7] = 1


    d1, d2 = model(quat, trans, pc, label_one_hot)
    print(d1)
    print(d2)

    # pc = get_real_pc(cat=args.cat, trial=args.trial)
    # pc = get_real_pc(cat='hammer', trial=3)
    # pc -= np.mean(pc, axis=0)
    # pc[:, 1] -=0.15

    # print(pc.shape)
    
    # print(pc.shape)
    # data = torch.from_numpy(pc.astype(np.float32))

    # # print(data.size())

    # seg = torch.zeros((1,2048)) + 7
    # # print(f"seg: {seg}")
    # # exit()

    # # seg = seg - seg_start_index

    # label = torch.Tensor([[7]]) # 7 - knife, 11 - mug
    # # label = torch.Tensor([[11]]) # 7 - knife, 11 - mug
    # label_one_hot = np.zeros((1, 16))
    # label_one_hot[0,7] = 1
    # label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))

    # # label_one_hot = np.zeros((label.shape[0], 16))
    # # for idx in range(label.shape[0]):
    # #     label_one_hot[idx, label[idx]] = 1
    # # label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))



    # data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
    # data = data.permute(0, 2, 1)
    # batch_size = data.size()[0]
    # print(data.size())
    # print(label_one_hot.size())
    # seg_pred = model(data, label_one_hot)
    # print(seg_pred)
    # print(seg_pred.shape)
    
    # exit()
    # seg_pred = seg_pred.permute(0, 2, 1).contiguous()
    # visiualization
    # np.savez_compressed('/home/gpupc2/GRASP/grasper/pat_semantic/hammer', data=data.cpu(), pred=pred.cpu())




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--cat', type=str, default='black_spatula',
                        help='category')
    parser.add_argument('--trial', type=int, default='1',
                        help='trial')
    # parser.add_argument('--visu', type=str, default='',
    #                     help='visualize the model')
    # parser.add_argument('--visu_format', type=str, default='ply',
    #                     help='file format of visualization')
    parser.add_argument('--seg_num', type=int, default=50,
                        help='Num of parts to segment')
    args = parser.parse_args()


    args.cuda = torch.cuda.is_available()
    test(args)