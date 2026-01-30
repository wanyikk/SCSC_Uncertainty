import torch.optim as optim
from net.network import WITT
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision
from tqdm import tqdm

import matplotlib.pyplot as plt
import os

import torch
print(torch.cuda.is_available())

parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='Kodak',
                    choices=['kodak', 'CLIC21'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=96,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
args = parser.parse_args()

class config():
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]

    filename2 = str.replace(filename, ':', '-')

    workdir = "Y:/teacherguo/Semantic Communications/WITT-main_02/history/{}".format(filename2)
    log = workdir + '/Log_{}.log'.format(filename2)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    #这个应该是训练轮数，原来为tot_epoch = 10000000
    tot_epoch = 100
    # if args.trainset == 'CIFAR10':
    #     save_model_freq = 5
    #     image_dims = (3, 32, 32)
    #     train_data_dir = "/media/Dataset/CIFAR10/"
    #     test_data_dir = "/media/Dataset/CIFAR10/"
    #     batch_size = 128
    #     downsample = 2
    #     encoder_kwargs = dict(
    #         img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
    #         embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
    #         window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #         norm_layer=nn.LayerNorm, patch_norm=True,
    #     )
    #     decoder_kwargs = dict(
    #         img_size=(image_dims[1], image_dims[2]),
    #         embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
    #         window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #         norm_layer=nn.LayerNorm, patch_norm=True,
    #     )
    # el
    if args.trainset == 'DIV2K':
        print(args.trainset)
        #标记求余除数,默认100。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
        save_model_freq = 100

        image_dims = (3, 256, 256)
        train_data_dir = ["Y:/teacherguo/Semantic Communications/WITT-main_02/DIV2K/DIV2K_train_HR"]

        if args.testset == "kodak":
            print("kodak数据集")
            test_data_dir = ["Y:/teacherguo/Semantic Communications/WITT-main_02/kodak灰度高清/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/Dataset/CLIC21/"]
        batch_size = 4
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained

def save_visualization(original, reconstructed, title='', save_path=''):
    """可视化原始图像和重构图像并保存到文件的函数"""
    #创建图像显示窗口
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # 将图像裁剪到有效范围
    original = original.clamp(0, 1)
    reconstructed = reconstructed.clamp(0, 1)

    # 显示原始图像
    axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # 显示重构图像
    axs[1].imshow(reconstructed.permute(1, 2, 0).cpu().numpy())  #reconstructed:是一个 PyTor度。ch 张量，表示模型重构的图像。其形状为 (C, H, W)，其中 C 是通道数（例如，RGB 图像的通道数为 3），H 是图像高度，W 是图像宽,
    #permute函数用于改变张量的维度顺序。(1, 2, 0)表示将原来的第1维（高度）移到第0维，将原来的第2维（宽度）移到第1维，将原来的第0维（通道）移到第2维。
    #.cpu()将张量从 GPU 转移到 CPU，因为 matplotlib 只处理 CPU 上的数据。
    #.numpy():将张量转换为NumPy数组。matplotlib的imshow函数需要NumPy数组作为输入，而不是PyTorch张量。
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_visualization_only_over(reconstructed, save_path=''):
    """只保存生成的图像"""
    reconstructed = reconstructed.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    plt.imsave(save_path, reconstructed)


def train_one_epoch(args):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        print("sssssss")
    #     for batch_idx, (input, label) in enumerate(train_loader):
    #         start_time = time.time()
    #         global_step += 1
    #         input = input.cuda()
    #         recon_image, CBR, SNR, mse, loss_G = net(input)
    #         loss = loss_G
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         elapsed.update(time.time() - start_time)
    #         losses.update(loss.item())
    #         cbrs.update(CBR)
    #         snrs.update(SNR)
    #         if mse.item() > 0:
    #             psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
    #             psnrs.update(psnr.item())
    #             msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
    #             msssims.update(msssim)
    #         else:
    #             psnrs.update(100)
    #             msssims.update(100)
    #
    #         if (global_step % config.print_step) == 0:
    #             process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
    #             log = (' | '.join([
    #                 f'Epoch {epoch}',
    #                 f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
    #                 f'Time {elapsed.val:.3f}',
    #                 f'Loss {losses.val:.3f} ({losses.avg:.3f})',
    #                 f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
    #                 f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
    #                 f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
    #                 f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
    #                 f'Lr {cur_lr}',
    #             ]))
    #             logger.info(log)
    #             for i in metrics:
    #                 i.clear()
    else:
        print("4==================================================================================")
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}'), 0):
            print(batch_idx, data)
            start_time = time.time()
            global_step += 1
            data = data.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(data)   #核心代码入口  data（B，3，256，256）
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - loss_G
                msssims.update(msssim)

            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()

def test():

    config.isTrain = False   # 设置训练模式为假
    net.eval()   # 将网络设置为评估模式
    # 初始化度量指标
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    # print("test01")

    for i in range(len(multiple_snr)):
        print("当前的i值")
        print(i)

        # print("test02")
        # 初始化结果数组
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):      # 遍历每个 SNR 值
        with (torch.no_grad()):        # 禁用梯度计算
            if args.trainset == 'CIFAR10':
                print("==============================testCLIC2021")
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}()',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
            else:
                print("==============================testkOADK==================")
                t = 0
                bps_sum=0
                bps_avg=0
                t1 = 0
                elapsed_sum=0
                elapsed_avg=0
                # print(test_loader)
                # print(len(test_loader))
                # print(len(train_loader))

                # for i, data in enumerate(train_loader):
                #     # 从data中解包出四个子张量，代表图像的四个通道
                #     channel1, channel2, channel3, channel4 = data
                #     if i < 1:  # 只打印前三个数据点          这里没问题，是DIV2K中的图片生成的图像张量
                #         print(channel1)
                #         print(channel2)
                #         print(channel3)
                #         print(channel4)
                for batch_idx, data in enumerate(test_loader):
                    # print(data)
                    # print("循环执行")
                    #if t == 0:
                     #   print(data)
                    start_time = time.time()
                    data = data.cuda()       # 将数据移动到 GPU
                    recon_image, CBR, SNR, mse, loss_G = net(data, SNR)   # 执行前向传播
                    elapsed.update(time.time() - start_time)   # 更新耗时

                    #计算带宽
                    data_size_in_bits = data.element_size() * data.nelement() * 8  # 将字节数转换为比特数
                    #print(data_size_in_bits)

                    bps = data_size_in_bits / elapsed.val

                    t = t+1
                    bps_sum = bps_sum + bps





                    cbrs.update(CBR)   # 更新 CBR
                    snrs.update(SNR)   # 更新 SNR
                    if mse.item() > 0:
                        # print("定义psnr")
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())   # 更新 PSNR
                        msssim = 1 - CalcuSSIM(data, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)    # 更新 MS-SSIM

                    else:
                        psnrs.update(100)
                        msssims.update(100)
                    # 记录日志
                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                        f'bps{bps:.4f}',
                    ]))
                    logger.info(log)
                    # 可视化第一批次的图像
                    #if batch_idx == 0:
                    save_path = os.path.join(config.samples, f'OtherDataset_SNR_{SNR}_batch_{batch_idx}.png')
                    # 检查输入图像和重构图像的尺寸
                    #print(f"Input image size: {data[0].shape}")

                   # print(f"Reconstructed image size: {recon_image[0].shape}")

                    save_visualization(data[0], recon_image[0], title=f'SNR: {SNR}', save_path=save_path)
                    save_visualization_only_over(recon_image[0], save_path=save_path)
                bps_avg = bps_sum / t

                print("当前平平均bps：", bps_avg)

                t = 0
                bps_sum = 0
                bps_avg = 0

        # 存储当前 SNR 的平均结果
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()      # 清除度量指标
    # 打印最终结果
    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

if __name__ == '__main__':

    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = WITT(args, config)
    # print(net)
    model_path = "WITT_AWGN_DIV2K_random_snr_psnr_C96.model"
    # print('02.' + model_path)

    load_weights(model_path)    #加入模型权重

    net = net.cuda()

    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    print('04.', end=' ')
    print(model_params)
    train_loader, test_loader = get_loader(args, config)
    print('05', end=' ')
    print(train_loader)
    print('06', end=' ')
    print(test_loader)
    cur_lr = config.learning_rate
    print('07', end=' ')
    print(cur_lr)
    optimizer = optim.Adam(model_params, lr=cur_lr)
    print('08', end=' ')
    print(optimizer)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    print('09', end=' ')
    print(steps_epoch)
    if args .training:
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)    #核心代码
            print("我是训练的标志===================轮数为：")
            print(epoch)
            if (epoch + 1) % config.save_model_freq == 0:
                save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename2, epoch + 1))
                test()
    else:
        print("调用test（）===================")
        test()

