import os
import numpy as np
import importlib
import argparse
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
#import Generative_spike_net as GSN
import RR
#from torchstat import stat
from collections import OrderedDict
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
def gisi_transform(spk, bz, t, wins, h, w):
    tf_mid = np.ones((bz, t, h, w)) * -1
    tf_right = np.ones((bz, t, h, w)) * -1
    tb_mid = np.ones((bz, t, h, w)) * t * wins
    tb_left = np.ones((bz, t, h, w)) * t * wins
    gisi = np.ones((bz, t, h, w))
    for i in range(0, t):
        for j in range(wins):
            index = spk[:, i, j, :, :] >= 1
            (tf_right[:, i, :, :])[index] = i * wins + j
            if j < 20:
                (tf_mid[:, i, :, :])[index] = i * wins + j
        if i < t - 1:
            tf_mid[:, i + 1, :, :] = copy.deepcopy(tf_right[:, i, :, :])
            tf_right[:, i + 1, :, :] = copy.deepcopy(tf_right[:, i, :, :])
    for i in range(t-1, -1, -1):
        for j in range(wins - 1, -1, -1):#search back
            index = spk[:, i, j, :, :] >= 1
            (tb_left[:, i, :, :])[index] = i * wins + j
            if j >= 20:
                (tb_mid[:, i, :, :])[index] = i * wins + j
        if i > 0:
            tb_mid[:, i - 1, :, :] = copy.deepcopy(tb_left[:, i, :, :])
            tb_left[:, i - 1, :, :] = copy.deepcopy(tb_left[:, i, :, :])
    for i in range(t):
        gisi[:, i, :, :] = tb_mid[:, i, :, :] - tf_mid[:, i, :, :]
    return gisi

def RawToSpike(video_seq, h, w, rec_frame, wins, spk_num):
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(video_seq)//(img_size//8)
    img_num = wins * spk_num
    SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0,h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8
    cnt = 0
    for img_id in range(rec_frame - int(img_num / 2), rec_frame + int(img_num / 2) + 1):
        id_start = img_id*img_size//8
        id_end = id_start + img_size//8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        SpikeMatrix[cnt, :, :] = (result == comparator)
        cnt += 1
    return SpikeMatrix, img_num
def strip_prefix(state_dict, prefix='module.'):
    if not all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    stripped_state_dict = {}
    for key in list(state_dict.keys()):
        stripped_state_dict[key.replace(prefix, '')] = state_dict.pop(key)
    return stripped_state_dict
def batch_metric(img, imclean, data_range = 255, w = 128, h = 128):
    #img = torch.clamp(img, 0, 1)
    #imclean = torch.clamp(imclean, 0, 1)
    img = (img * 255.0).astype(np.uint8)
    Iclean = (imclean * 255.0).astype(np.uint8)
    Iclean = Iclean.reshape(-1, h, w)
    img = img.reshape(-1, h, w)
    #Iclean = np.transpose(Iclean, (1, 2, 0))
    #img = np.transpose(img, (1, 2, 0))
    PSNR = compare_psnr(img[0,:,:], Iclean[0,:,:], data_range=data_range)
    #print(img.shape)
    SSIM = compare_ssim(img, Iclean, data_range=data_range, channel_axis = 0)
    #print("psnr:", PSNR, "  ssim:", SSIM)
    return PSNR, SSIM
def test(Dataset, Net, Pre_model = 0, End_frame = 0, Tot_class = 10, W = 1000, H = 1000, Kinds = 2, Save = "", Frame_interval = 41, Data = ""):
    print("begin!!!")
    save_dir = Save
    temp_loss = 0
    avg_psnr = 0
    avg_ssim = 0
    tot_class = 0
    avg_psnr_scale = 0
    file_path =  Data
    wins = Frame_interval
    spk_num = 21
    rec_frame = 300 #重建中间帧数
    for i in class_name:
        input_f = open(file_path, 'rb+')
        video_seq = input_f.read()
        video_seq = np.fromstring(video_seq, 'B')
        spks, valid_frame = RawToSpike(video_seq, H, W, rec_frame, wins, spk_num)
        spks = spks.reshape(spk_num, wins, H, W)
        spks = np.expand_dims(spks, 0)
        tfis = gisi_transform(spks, 1, spk_num, wins, H, W)
        os.makedirs(os.path.join(save_dir, "imgs", i), exist_ok = True)
        tfis = torch.tensor(tfis).float()
        spks = torch.tensor(spks).float()
        with torch.no_grad():
            img_final = Net(spks.cuda(), tfis.cuda(), True)
        max_save = img_final.shape[1]
        for save_i in range(max_save):#
            save_img_final = torch.clamp(img_final[0,save_i,0,0:H,:].detach(), 0, 1).cpu().numpy()
            save_img_final = ((save_img_final)**(1/2.2))
            save_img_final[save_img_final > 1] = 1
            save_img_final = (save_img_final * 255).astype(np.uint8)
            save_id_class = i
            save_id_frame = (save_i - int(max_save / 2)) * 41 + rec_frame
            if os.path.exists(os.path.join(save_dir, "imgs")) == False:
                os.mkdir(os.path.join(save_dir, "imgs"))
            if os.path.exists(os.path.join(save_dir, "imgs", str(save_id_class))) == False:
                os.mkdir(os.path.join(save_dir, "imgs", str(save_id_class)))
            cv2.imwrite(os.path.join(save_dir, "imgs", str(save_id_class), str(save_id_frame) + "_final.png"), save_img_final)
        tot_class += 1
    return
def get_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1010, help='Number of epochs')
    parser.add_argument('--bz', '-b', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    #parser.add_argument('--decay', '-d', type=float, default=0.3, help='decay of Learning rate')
    parser.add_argument('--kinds', '-k', type = int, default=1, help='train or test')
    parser.add_argument('--frame', '-f', type = int, default=20, help='the number of spk stream for a scene')
    parser.add_argument('--train_model', '-tm', type = str, default="0", help='load model for train')
    parser.add_argument('--test_model', '-m', type = str, default="100", help='load model for test')
    parser.add_argument('--gpu', '-g', type = str, default="0", help='gpu_id')
    parser.add_argument('--wins', '-w', type = int, default=41, help='wins')
    parser.add_argument('--feat_num', '-fn', type = int, default=64, help='feat num')
    parser.add_argument('--block_num', '-bn', type = int, default=16, help='block num')
    parser.add_argument('--crop', '-c', type=int, default=64, help='crop')
    parser.add_argument('--is_prop', '-pr', type=bool, default=True, help='is_prop')
    parser.add_argument('--is_multi', '-mu', type=bool, default=False, help='is_multi')
    parser.add_argument('--frame_interval', '-fi', type=int, default=41, help='frame_interval')
    parser.add_argument('--t', '-t', type=int, default=21, help='tot time')
    parser.add_argument('--tot_class', '-to', type=int, default=100, help='tot calss')
    parser.add_argument('--init_frame', '-if', type=int, default=20, help='tot time')
    parser.add_argument('--save_path', '-sp', type=str, default="", help='save_path')
    parser.add_argument('--data_path', '-dp', type=str, default="", help='data_path')
    return parser.parse_args()
def dynamic_import(module):
    return importlib.import_module(module)
if __name__=="__main__":
    args = get_args()
    if args.is_multi == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"
        device_ids = [0, 1]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    loss = torch.nn.L1Loss()
    net = RR.Net(wins = args.wins, feat_num = args.feat_num, block_num = args.block_num).cuda()
    if args.is_multi == True:
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    print("test start!")
    save_path = args.save_path
    data_path = args.data_path
    checkpoint = torch.load("./model.pth")
    checkpoint = strip_prefix(checkpoint)
    net.load_state_dict(checkpoint['net_dict'])
    test(Dataset = None, Net = net.eval(), Kinds = args.kinds, Save = save_path, Frame_interval = args.frame_interval, Data = data_path)