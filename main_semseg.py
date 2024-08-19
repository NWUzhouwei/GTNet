#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_semseg.py
@Time: 2021/7/20 7:49 PM
"""
#python main_semseg.py --exp_name=semseg_5_1 --test_area=5
#python main_semseg.py --exp_name=semseg_5_1_lr0.1_dp0.5_withoutgrad_k10 --test_area=5
#python main_semseg.py --exp_name=semseg_5_1_lr0.01_dp0.5_k20_layer3 --test_area=5
#python main_semseg.py --exp_name=semseg_5_1_lr0.1_dp0.5_k20_layer4 --test_area=5
from __future__ import print_function
import os

import argparse
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numba
import h5py
import math
from tensorboardX import SummaryWriter
from data import S3DIS,S3DISDataset
from model import GTNet_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss,IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement

global room_seg
room_seg = []
global room_pred
room_pred = []
global visual_warning
visual_warning = True

s3dis_color_classes = {0:	[0,250,129],
                 1:	[62,188,202],
                 2:	[119,52,96],
                 3:       [118,77,57],
                 4:      [98,65,24],
                 5:      [176,199,220],
                 6:        [219,69,32],
                 7:       [101,147,74],
                 8:      [0,250,120],
                 9:        [1,165,175],
                 10:    [246,139,80],
                 11:       [229,187,129],
                 12:     [205,201,125],
                 13:      [255,255,255]}

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs\\' + args.exp_name):
        os.makedirs('outputs\\' + args.exp_name)
    if not os.path.exists('outputs\\' + args.exp_name + '\\' + 'models'):
        os.makedirs('outputs\\' + args.exp_name + '\\' + 'models')
    # os.system('cp main_semseg.py outputs' + '/' + args.exp_name + '/' + 'main_semseg.py.backup')
    # os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    # os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    # os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')
    # os.system('cp transformer.py outputs' + '/' + args.exp_name + '/' + 'transformer.py.backup')
    global writer
    writer= SummaryWriter('outputs\\%s\\models\\' % args.exp_name)

def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(13):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1
    return I_all / U_all


def visualization(visu, visu_format, test_choice, data, seg, pred, visual_file_index, semseg_colors):
    global room_seg, room_pred
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        with open("D:\\datasets\\indoor3d_sem_seg_hdf5_data_test\\room_filelist.txt") as f:
            files = f.readlines()
            test_area = files[visual_file_index][5]
            roomname = files[visual_file_index][7:-1]
            if visual_file_index + 1 < len(files):
                roomname_next = files[visual_file_index + 1][7:-1]
            else:
                roomname_next = ''
        if visu[0] != 'all':
            if len(visu) == 2:
                if visu[0] != 'area' or visu[1] != test_area:
                    skip = True
                else:
                    visual_warning = False
            elif len(visu) == 4:
                if visu[0] != 'area' or visu[1] != test_area or visu[2] != roomname.split('_')[0] or visu[3] != \
                        roomname.split('_')[1]:
                    skip = True
                else:
                    visual_warning = False
            else:
                skip = True
        elif test_choice != 'all':
            skip = True
        else:
            visual_warning = False
        if skip:
            visual_file_index = visual_file_index + 1
        else:
            if not os.path.exists(
                    'outputs\\' + args.exp_name + '\\' + 'visualization' + '\\' + 'area_' + test_area + '\\' + roomname):
                os.makedirs(
                    'outputs\\' + args.exp_name + '\\' + 'visualization' + '\\' + 'area_' + test_area + '\\' + roomname)

            data = np.loadtxt(
                'D:\\datasets\\indoor3d_sem_seg_hdf5_data_test\\raw_data3d\\Area_' + test_area + '\\' + roomname + '(' + str(
                    visual_file_index) + ').txt')
            visual_file_index = visual_file_index + 1
            for j in range(0, data.shape[0]):
                RGB.append(semseg_colors[int(pred[i][j])])
                RGB_gt.append(semseg_colors[int(seg[i][j])])
            data = data[:, [1, 2, 0]]
            xyzRGB = np.concatenate((data, np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((data, np.array(RGB_gt)), axis=1)
            room_seg.append(seg[i].cpu().numpy())
            room_pred.append(pred[i].cpu().numpy())
            f = open(
                'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '.txt',
                "a")
            f_gt = open(
                'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_gt.txt',
                "a")
            np.savetxt(f, xyzRGB, fmt='%s', delimiter=' ')
            np.savetxt(f_gt, xyzRGB_gt, fmt='%s', delimiter=' ')

            if roomname != roomname_next:
                mIoU = np.mean(calculate_sem_IoU(np.array(room_pred), np.array(room_seg), visual=True))
                mIoU = str(round(mIoU, 4))
                room_pred = []
                room_seg = []
                if visu_format == 'ply':
                    filepath = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_pred_' + mIoU + '.ply'
                    filepath_gt = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_gt.ply'
                    xyzRGB = np.loadtxt(
                        'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '.txt')
                    xyzRGB_gt = np.loadtxt(
                        'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_gt.txt')
                    xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i
                              in range(xyzRGB.shape[0])]
                    xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4],
                                  xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                    vertex = PlyElement.describe(np.array(xyzRGB,
                                                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                                 ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath)
                    print('PLY visualization file saved in', filepath)
                    vertex = PlyElement.describe(np.array(xyzRGB_gt,
                                                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                                 ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath_gt)
                    print('PLY visualization file saved in', filepath_gt)
                    os.system(
                        'rm -rf ' + 'outputs/' + args.exp_name + '/visualization/area_' + test_area + '/' + roomname + '/*.txt')
                else:
                    filename = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '.txt'
                    filename_gt = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_gt.txt'
                    filename_mIoU = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_pred_' + mIoU + '.txt'
                    os.rename(filename, filename_mIoU)
                    print('TXT visualization file saved in', filename_mIoU)
                    print('TXT visualization file saved in', filename_gt)
            elif visu_format != 'ply' and visu_format != 'txt':
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                      (visu_format))
                exit()

import random
def worker_init_fn(worker_id):
    random.seed(1 + worker_id)

def train(args, io):
    root = 'D:\\datasets\\s3dis'

    train_loader = DataLoader(S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area),
                              num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True, worker_init_fn = worker_init_fn)
    test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area),
                             num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=False, worker_init_fn =worker_init_fn)
    
    # train_loader = DataLoader(S3DISDataset(root = root, num_points=args.num_points, split='train', with_normalized_coords=True, holdout_area=args.test_area),
    #                           num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(S3DISDataset(root = root, num_points=args.num_points, split='test', with_normalized_coords=True, holdout_area=args.test_area),
    #                          num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'GTNet':
        model = GTNet_semseg(args).to(device)
    else:
        raise Exception("Not implemented")

    
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
        # steplr opt
        # opt = optim.SGD([{'params':model.parameters(),'initial_lr':args.lr*100}], lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    try:
      print("load model*******************************")
      checkpoint = torch.load('outputs\\%s\\models\\best_model_%s.pth' % (args.exp_name, args.test_area))
      start_epoch = checkpoint['epoch']
      # print(start_epoch)
      model.load_state_dict(checkpoint['model_state_dict'])
      if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
          print(' => loading optimizer')
          opt.load_state_dict(checkpoint['optimizer_state_dict'])
          # print(start_epoch)
    except:
        start_epoch = 0
    print(start_epoch)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.1, args.epochs)

    criterion = cal_loss

    best_test_iou = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        ####################
        # Train
        # https://github.com/pytorch/pytorch/issues/69611
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        # with autograd.detect_anomaly():
        for data, seg in tqdm(train_loader, total=len(train_loader)):
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            # torch.cuda.empty_cache()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)
        writer.add_scalar('loss_train', train_loss*1.0/count, epoch)
        writer.add_scalar('mIoU_train', np.mean(train_ious), epoch)
        writer.add_scalar('mAcc_train', avg_per_class_acc, epoch)
        writer.add_scalar('allAcc_train', train_acc, epoch)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        with torch.no_grad():
            for data, seg in tqdm(test_loader, total=len(test_loader)):
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)

        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss * 1.0 / count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        writer.add_scalar('loss_val', test_loss*1.0/count, epoch)
        writer.add_scalar('mIoU_val', np.mean(test_ious), epoch)
        writer.add_scalar('mAcc_val', avg_per_class_acc, epoch)
        writer.add_scalar('allAcc_val', test_acc,epoch)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            state = {
                'epoch': epoch,
                'iou': best_test_iou,
                'acc': test_acc,
                'avg_per_class_acc': avg_per_class_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }

            torch.save(state, 'outputs/%s/models/best_model_%s.pth' % (args.exp_name, args.test_area))
        latest_state={
                'epoch': epoch,
                'iou': np.mean(test_ious),
                'acc': test_acc,
                'avg_per_class_acc': avg_per_class_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }
        torch.save(latest_state, 'outputs/%s/models/latest_model_%s.pth' % (args.exp_name, args.test_area))


@numba.jit()
def update_scene_predictions(batched_confidences, batched_predictions, batched_shuffled_point_indices,
                             scene_confidences, scene_predictions, window_to_scene_mapping, total_num_voted_points,
                             batch_size, min_window_index):
    for b in range(batch_size):
        window_index = min_window_index + b
        current_window_mapping = window_to_scene_mapping[window_index]
        current_shuffled_point_indices = batched_shuffled_point_indices[b]
        current_confidences = batched_confidences[b]
        current_predictions = batched_predictions[b]
        for p in range(total_num_voted_points):
            point_index = current_window_mapping[current_shuffled_point_indices[p]]
            current_confidence = current_confidences[p]
            if current_confidence > scene_confidences[point_index]:
                scene_confidences[point_index] = current_confidence
                scene_predictions[point_index] = current_predictions[p]

@numba.jit()
def update_stats(stats, ground_truth, predictions, scene_index, total_num_points_in_scene):
    for p in range(total_num_points_in_scene):
        gt = int(ground_truth[p]) # 第p个点属于哪个类别
        pd = int(predictions[p])
        stats[0, gt, scene_index] += 1 # 真实值
        stats[1, pd, scene_index] += 1 # 预测值
        if gt == pd:
            stats[2, gt, scene_index] += 1 # 交叉部分
    gt_total = stats[0].sum()
    pd_true_total = stats[2].sum()
    return pd_true_total / gt_total
import logging
   
macc_list = []
miou_list = []
def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    iou=[]
    def log_string(str):
        logger.info(str)
        print(str)
     # 创建一个Logger
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.INFO)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(os.path.join("outputs\\semseg", 'test_all.log'))
    fh.setLevel(logging.INFO)
    # 定义hander的输出格式（formatter）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 给handler添加formatter
    fh.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    # log_string('PARAMETER ...')
    # log_string(configs)
   
    def print_stats(stats):
    
        iou = stats[2] / (stats[0] + stats[1] - stats[2])
        log_string('classes: {}'.format('  '.join(map('{:>8d}'.format, stats[0].astype(np.int64)))))
        log_string('positiv: {}'.format('  '.join(map('{:>8d}'.format, stats[1].astype(np.int64)))))
        log_string('truepos: {}'.format('  '.join(map('{:>8d}'.format, stats[2].astype(np.int64)))))
        log_string('clssiou: {}'.format('  '.join(map('{:>8.2f}'.format, iou * 100))))
        macc = stats[2].sum() / stats[1].sum()
        miou = iou.mean()
        macc_list.append(macc)
        miou_list.append(miou)
        log_string('meanAcc: {:4.2f}'.format( macc * 100))
        log_string('meanIoU: {:4.2f}'.format( miou * 100))

    def print_stats_all(stats):
        iou = stats[2] / (stats[0] + stats[1] - stats[2])
        log_string('6次，全部点的结果：')
        log_string('meanAcc_all: {:4.2f}'.format(stats[2].sum() / stats[1].sum() * 100))
        log_string('meanIoU_all: {:4.2f}'.format(iou.mean() * 100))
        log_string('6次结果的平均：')
        log_string('meanAcc_avg: {:4.2f}'.format(np.mean(macc_list) * 100))
        log_string('meanIoU_avg: {:4.2f}'.format(np.mean(miou_list) * 100))
    # stats_list = []
    stats_all=np.zeros((3,13))
    test_area=args.test_area
    # for test_area in range(1, 7):
    visual_file_index = 0
    stats_path = 'outputs\\semseg\\semseg_5_%s\\best.eval.npy' % test_area
    # if os.path.exists(stats_path):
    #     stats = np.load(stats_path)
    #     stats = stats.sum(axis=-1)
    #     log_string("\n==>model_%d evaluate outcome:" % test_area)
    #     print_stats(stats)
    #     stats_all = stats_all + stats
    #     continue
    log_string(f'==>model_"{test_area}" testing')
    test_area = str(test_area)
    # if os.path.exists("D:\\datasets\\indoor3d_sem_seg_hdf5_data_test\\room_filelist.txt"):
    #     with open("D:\\datasets\\indoor3d_sem_seg_hdf5_data_test\\room_filelist.txt") as f:
    #         for line in f:
    #             if (line[5]) == test_area:
    #                 break
    #             visual_file_index = visual_file_index + 1
    # if (args.test_area == 'all') or (test_area == args.test_area):
        # test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
        #                          batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    root = 'D:\\datasets\\s3dis'
    dataset=S3DISDataset(root = root, num_points=args.num_points, split='test', with_normalized_coords=True, holdout_area=test_area)
                    
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    # semseg_colors = test_loader.dataset.semseg_colors
    if args.model == 'GTNet':
        model = GTNet_semseg(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(
        torch.load('semseg\\semseg_5_%s\\models\\best_model_%s.pth'%(test_area,test_area))['model_state_dict'])
    model = model.eval()
    # 打印模型的 state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total_num_scenes = len(dataset.scene_list)
    stats = np.zeros((3, 13, total_num_scenes))
    visual_dir = 'outputs\\semseg\\visualization\\'
    # visual_dir = Path(visual_dir)
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    for scene_index, (scene, scene_files) in enumerate(tqdm(dataset.scene_list.items(), desc='eval', ncols=0)):
        ground_truth = np.load(os.path.join(scene, 'label.npy')).reshape(-1)
        whole_scene_data = np.load(os.path.join(scene, 'xyzrgb.npy')) 
        total_num_points_in_scene = ground_truth.shape[0]
        confidences = np.zeros(total_num_points_in_scene, dtype=np.float32)
        predictions = np.full(total_num_points_in_scene, -1, dtype=np.int64)

        for filename in scene_files:
            h5f = h5py.File(filename, 'r')
            scene_data = h5f['data'][...].astype(np.float32)
            scene_num_points = h5f['data_num'][...].astype(np.int64)
            window_to_scene_mapping = h5f['indices_split_to_full'][...].astype(np.int64)

            num_windows, max_num_points_per_window, num_channels = scene_data.shape
            extra_batch_size = 1 * math.ceil(max_num_points_per_window / dataset.num_points)
            total_num_voted_points = extra_batch_size * dataset.num_points

            for min_window_index in range(0, num_windows,args.test_batch_size):
                max_window_index = min(min_window_index + args.test_batch_size, num_windows)
                batch_size = max_window_index - min_window_index
                window_data = scene_data[np.arange(min_window_index, max_window_index)]
                window_data = window_data.reshape(batch_size, -1, num_channels)

                # repeat, shuffle and tile
                # TODO: speedup here
                batched_inputs = np.zeros((batch_size, total_num_voted_points, num_channels), dtype=np.float32)
                batched_shuffled_point_indices = np.zeros((batch_size, total_num_voted_points), dtype=np.int64)
                for relative_window_index in range(batch_size):
                    num_points_in_window = scene_num_points[relative_window_index + min_window_index]
                    num_repeats = math.ceil(total_num_voted_points / num_points_in_window)
                    shuffled_point_indices = np.tile(np.arange(num_points_in_window), num_repeats)
                    shuffled_point_indices = shuffled_point_indices[:total_num_voted_points]
                    np.random.shuffle(shuffled_point_indices)
                    batched_shuffled_point_indices[relative_window_index] = shuffled_point_indices
                    batched_inputs[relative_window_index] = window_data[relative_window_index][shuffled_point_indices]

                # model inference
                inputs = torch.from_numpy(
                    batched_inputs.reshape((batch_size * extra_batch_size, dataset.num_points, -1))
                ).float().to(device)
                with torch.no_grad(): # model test
                    batched_confidences, batched_predictions = (model(inputs).permute(0, 2, 1).contiguous()).max(dim=2)

                    batched_confidences = batched_confidences.view(batch_size, total_num_voted_points).cpu().numpy()
                    batched_predictions = batched_predictions.view(batch_size, total_num_voted_points).cpu().numpy()

                update_scene_predictions(batched_confidences, batched_predictions, batched_shuffled_point_indices,
                                        confidences, predictions, window_to_scene_mapping,
                                        total_num_voted_points, batch_size, min_window_index)

        # update stats
        iou=update_stats(stats, ground_truth, predictions, scene_index, total_num_points_in_scene)
        # print(iou)
        fout = open(os.path.join(visual_dir, str(scene_index) + '_pred.obj'), 'w')
        fout_gt = open(os.path.join(visual_dir, str(scene_index) + '_gt.obj'), 'w')

        filename_visual = os.path.join(visual_dir, str(scene_index) + '.txt')
        with open(filename_visual, 'w') as pl_save:
            for pre_i in predictions:
                pl_save.write(str(int(pre_i)) + '\n')
            pl_save.close()
        filename_iou = os.path.join(visual_dir, str(scene_index) + 'iou' + str(format(iou, '.4f')) + '.txt')
        with open(filename_iou, 'w') as single_iou_save:
            single_iou_save.write(str(iou))
            single_iou_save.close()
        for i in range(ground_truth.shape[0]):
            if predictions[i] == -1:
                continue
            else:
                color = s3dis_color_classes[predictions[i]]
                    # color = g_label2color[predictions[i]]
            color_gt = s3dis_color_classes[ground_truth[i]]
            if predictions[i]!=1 and predictions[i]!=-1:
                fout.write('v %f %f %f %d %d %d\n' % (
                    whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                    color[2]))
                fout_gt.write(
                    'v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                        color_gt[1], color_gt[2]))
        fout.close()
        fout_gt.close()
    np.save(stats_path, stats)
    log_string(f"==>save best.eval.npy on '{stats_path}'")
    test_area=int(test_area)
    log_string("\n==>model_%d evaluate outcome:" % test_area)
    stats = stats.sum(axis=-1) # (3, 13)
    print_stats(stats)
    stats_all = stats_all + stats
        # stats_list.append(stats)
    # stats_list=torch.tensor(stats_list)
    # stats_all = torch.sum(torch.cat(stats_list, dim=-1), keepdim=False, dim=-1)
    print_stats_all(stats_all)
    #         test_acc = 0.0
    #         count = 0.0
    #         test_true_cls = []
    #         test_pred_cls = []
    #         test_true_seg = []
    #         test_pred_seg = []
            
    #         with torch.no_grad():
    #             for data, seg in tqdm(test_loader, total=len(test_loader)):
    #                 data, seg = data.to(device), seg.to(device)
    #                 data = data.permute(0, 2, 1)
    #                 batch_size = data.size()[0]
    #                 seg_pred = model(data)
    #                 seg_pred = seg_pred.permute(0, 2, 1).contiguous()
    #                 pred = seg_pred.max(dim=2)[1]
    #                 seg_np = seg.cpu().numpy()
    #                 pred_np = pred.detach().cpu().numpy()
    #                 test_true_cls.append(seg_np.reshape(-1))
    #                 test_pred_cls.append(pred_np.reshape(-1))
    #                 test_true_seg.append(seg_np)
    #                 test_pred_seg.append(pred_np)
    #                 # visiualization
    #                 # visualization(args.visu, args.visu_format, args.test_area, data, seg, pred, visual_file_index,
    #                 #               semseg_colors)
    #                 # visual_file_index = visual_file_index + data.shape[0]
    #         if visual_warning and args.visu != '':
    #             print('Visualization Failed: You can only choose a room to visualize within the scope of the test area')
    #         test_true_cls = np.concatenate(test_true_cls)
    #         test_pred_cls = np.concatenate(test_pred_cls)
    #         test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    #         avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    #         test_true_seg = np.concatenate(test_true_seg, axis=0)
    #         test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    #         test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
    #         outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
    #                                                                                                 test_acc,
    #                                                                                                 avg_per_class_acc,
    #                                                                                                 np.mean(test_ious))
    #         io.cprint(outstr)
    #         all_true_cls.append(test_true_cls)
    #         all_pred_cls.append(test_pred_cls)
    #         all_true_seg.append(test_true_seg)
    #         all_pred_seg.append(test_pred_seg)
    #         iou.append(np.mean(test_ious))
    # if args.test_area == 'all':
    #     all_true_cls = np.concatenate(all_true_cls)
    #     all_pred_cls = np.concatenate(all_pred_cls)
    #     all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
    #     avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
    #     all_true_seg = np.concatenate(all_true_seg, axis=0)
    #     all_pred_seg = np.concatenate(all_pred_seg, axis=0)
    #     all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
    #     outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
    #                                                                                      avg_per_class_acc,
    #                                                                                      np.mean(all_ious))
    #     io.cprint(outstr)
    #     outstr='6-fold::test iou:%.6f'%np.mean(iou)
    #     io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='semseg_all', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='GTNet', metavar='N',
                        choices=['GTNet'],
                        help='Model to use, [GTNet]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--test_area', type=str, default="5", metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
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
    parser.add_argument('--eval', type=bool, default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=15, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='D:\\wq\\GTNet.pytorch-master\\outputs\\semseg\\', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    _init_()


    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
