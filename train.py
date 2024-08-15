import argparse
import os
from math import log10

import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='训练超分辨率模型')
parser.add_argument('--crop_size', default=88, type=int, help='训练图像裁剪大小')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='超分辨率放大倍数')
parser.add_argument('--num_epochs', default=100, type=int, help='训练轮次')

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    train_set = TrainDatasetFromFolder('C:\\Users\\ads\\Desktop\\SRdata\\train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('C:\\Users\\ads\\Desktop\\SRdata\\val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    total_iterations = 200000
    lr_schedule_step = 50000
    initial_lr = 1e-4
    lambda_loss = 10
    eta = 5e-3
    gamma = 1e-6

    netG = Generator(UPSCALE_FACTOR)
    print('# 生成器参数数量:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# 判别器参数数量:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9, 0.99))

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data
            if torch.cuda.is_available():
                z = z.float().cuda()

            fake_img = netG(z).detach()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] 损失_D: %.4f 损失_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            max_val_hr = 0
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size

                if torch.cuda.is_available():
                    val_lr = val_lr.cuda()
                    val_hr = val_hr.cuda()

                sr = netG(val_lr)

                batch_mse = ((sr - val_hr) ** 2).data.mean()
                batch_ssim = pytorch_ssim.ssim(sr, val_hr).item()

                valing_results['mse'] += batch_mse * batch_size
                valing_results['ssims'] += batch_ssim * batch_size

                max_val_hr = max(max_val_hr, val_hr.max().item())

                val_images.extend([
                    display_transform()(val_hr_restore.squeeze(0)),
                    display_transform()(val_hr.squeeze(0)),
                    display_transform()(sr.squeeze(0))
                ])

                del val_lr, val_hr, sr, val_hr_restore
                torch.cuda.empty_cache()

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[保存训练结果]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

            valing_results['psnr'] = 10 * log10(
                (max_val_hr ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[将LR图像转换为SR图像] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))

        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            data_frame = pd.DataFrame(
                data={'损失_D': results['d_loss'], '损失_G': results['g_loss'], '得分_D': results['d_score'],
                      '得分_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='轮次')

        if epoch % (total_iterations // lr_schedule_step) == 0:
            for param_group in optimizerG.param_groups:
                param_group['lr'] *= 0.5
            for param_group in optimizerD.param_groups:
                param_group['lr'] *= 0.5
