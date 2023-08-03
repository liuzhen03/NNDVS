import argparse
import logging
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from image_warper import FlowWarper
from metrices import MetricAnalyzer
from model import PathSmoothUNet
from utils import getAllFileInDir, load_checkpoint
from dataset import NUSDataset

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='NUS_dataset', help="dataset name")
parser.add_argument('--dataset_dir', default='./data/', help="root dir of the dataset")
parser.add_argument('--video_dir', default='Regular/original/videoes', help="dir of the input videos")
parser.add_argument('--motion_dir', default='Regular/original/motions/mesh', help="dir of the input motions")
parser.add_argument('--result_dir', default='./experiments', help="save dir")
parser.add_argument('--concat_dir', default='concat', help="dir of the concat videos")

parser.add_argument('--model_path', default='./pretrained/best.pth.tar', help="pretrained model weights")
parser.add_argument('--net_radius', default=15, type=int, help="radius of the input historical frames")
parser.add_argument('--scale_factor', default=8, type=int, help="scale to resize the input flowmap")
parser.add_argument('--latency', default=0, type=int, help="latency number")
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--bs', default=1, type=int, help="batch size")
parser.add_argument('--con_num', default=1, type=int, help="number of continuous samples to merge")


def fetch_dataloaders():

    # fetch dataloaders
    logging.info("Fetch Evaluate Dataloaders...")
    eval_dataloaders = []
    for f in range(data_length):
        video_path = os.path.join(all_video_dir, "{}.mp4".format(f))
        capture = cv2.VideoCapture()
        capture.open(video_path)
        motion_path = os.path.join(all_motion_dir, "{}.npy".format(f))
        gt_path = None
        eval_dataset = NUSDataset([motion_path], gt_path, args.net_radius, args.con_num, 
                                                        2*args.net_radius - args.latency)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        eval_dataloaders.append(eval_dataloader)

    return eval_dataloaders


def get_warp_trans(video_idx, net, dataloader):

    video_path = os.path.join(all_video_dir, "{}.mp4".format(video_idx))
    capture = cv2.VideoCapture()
    capture.open(video_path)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bilinear_upsample = nn.Upsample(scale_factor=args.scale_factor, mode='bilinear', align_corners=True)

    net.eval()
    warp_trans = np.zeros((len(dataloader.dataset) + 2*args.net_radius + args.con_num - 1,
                          frame_height, frame_width, 2), 
                         dtype=np.float32)
    warp_trans_gt = np.zeros((len(dataloader.dataset) + 2*args.net_radius + args.con_num - 1,
                          frame_height, frame_width, 2), 
                         dtype=np.float32)
    run_num = 0
    loop = tqdm(dataloader, leave=False)

    for flows, gts in loop:
        flows = -flows.cuda()
        Bi = net(flows[:, 0, :, :, :])
        loop.set_description(f'Video: [{video_idx}/{data_length}]')
        Bi = bilinear_upsample(Bi)
        Bi = np.squeeze(Bi.detach().cpu().numpy().transpose(0, 2, 3, 1))
        warp_trans[(run_num + 2*args.net_radius - args.latency):(run_num + 2*args.net_radius - args.latency+flows.shape[0])] = Bi
        Bi_gt = -bilinear_upsample(gts[:, 0, :, :, :])
        Bi_gt = np.squeeze(Bi_gt.detach().cpu().numpy().transpose(0, 2, 3, 1))
        warp_trans_gt[(run_num + 2*args.net_radius - args.latency):(run_num + 2*args.net_radius - args.latency+flows.shape[0])] = Bi_gt
        run_num += flows.shape[0]
    
    return warp_trans, warp_trans_gt


def render_res_video(video_idx, warp_trans, warp_trans_gt):

    video_path = os.path.join(all_video_dir, "{}.mp4".format(video_idx))
    res_video_path = os.path.join(res_video_dir, "{}.mp4".format(video_idx))
    res_concat_path = os.path.join(res_concat_dir, "{}.mp4".format(video_idx))
    capture = cv2.VideoCapture()
    capture.open(video_path)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    image_warper = FlowWarper()
    image_warper.initialize(frame_width, frame_height)
    trans_length = min(frame_count, warp_trans.shape[0])

    writer = cv2.VideoWriter(res_video_path, fourcc, fps, (frame_width, frame_height))
    writer_concat = cv2.VideoWriter(res_concat_path, fourcc, fps, (frame_width*2, frame_height))
    real_trans_length = 0

    for i in range(trans_length):
        ok, frame = capture.read()
        if not ok:
            break
        new_frame = image_warper.warp_image(frame, warp_trans[i])
        writer.write(new_frame)
        cat_frame = utils.concatImagesHorizon([frame, new_frame])
        writer_concat.write(cat_frame)
        real_trans_length += 1
    metric_analyzer = MetricAnalyzer(frame_width, frame_height)

    return metric_analyzer


def eval_videos(net, dataloaders):

    crop_ratio = utils.RunningAverage()
    distortion_value = utils.RunningAverage()
    stability_score = utils.RunningAverage()
    for idx in range(data_length):
        logging.info("Eval Video: {}".format(idx))

        # get warp transformation
        warp_trans, warp_trans_gt = get_warp_trans(idx, net, dataloaders[idx])

        # render video and calculate metrics
        metric_analyzer = render_res_video(idx, warp_trans, warp_trans_gt)
        cr, dv, ss = metric_analyzer.run(warp_trans, os.path.join(res_video_dir, "{}.mp4".format(idx)))
        crop_ratio.update(cr)
        distortion_value.update(dv)
        stability_score.update(ss)

    logging.info("Avg Crop Ratio: {:05.3f}".format(crop_ratio()))
    logging.info("Avg Distortion Value: {:05.3f}".format(distortion_value()))
    logging.info("Avg Stability Score: {:05.3f}".format(stability_score()))
        

if __name__ == '__main__':

    args = parser.parse_args()

    # set the global pathes and variables
    data_dir = os.path.join(args.dataset_dir, args.dataset)
    all_video_dir = os.path.join(data_dir, args.video_dir)
    all_motion_dir = os.path.join(data_dir, args.motion_dir)
    all_video_pathes = getAllFileInDir(all_video_dir)
    data_length = len(all_video_pathes)

    res_video_dir = os.path.join(args.result_dir, args.video_dir.split('/')[0])
    res_concat_dir = os.path.join(res_video_dir, args.concat_dir)
    utils.checkAndMakeDir(res_video_dir)
    utils.checkAndMakeDir(res_concat_dir)

    # set logger
    utils.set_logger(os.path.join(res_video_dir, 'evaluate.log'))

    # load dataset
    logging.info("Loading Datasets...")
    logging.info("Video Dir: {}".format(all_video_dir))
    logging.info("Motion Dir: {}".format(all_motion_dir))
    eval_dataloaders = fetch_dataloaders()

    # load model
    logging.info("Loading Pretrained Model From: {}".format(args.model_path))
    net = PathSmoothUNet(4 * args.net_radius)
    net = nn.DataParallel(net)
    torch.backends.cudnn.benchmark = False
    net = net.cuda()
    load_checkpoint(args.model_path, net)
    
    # eval
    logging.info("Starting Evaluation...")
    eval_videos(net, eval_dataloaders)
    logging.info("Finished...")
