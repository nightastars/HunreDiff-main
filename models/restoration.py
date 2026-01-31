import torch
import torch.nn as nn
import utils
import torchvision
import os
from utils.measure import compute_measure
import numpy as np
import time
import matplotlib.pyplot as plt
import imageio
import SimpleITK as sitk
from utils.multiCTmain import FanBeam


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def diffusive_restoration(self, x_cond):
        xs = self.diffusion.sample_image(x_cond)
        return xs

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=-1024, vmax=2048)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=-1024, vmax=2048)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=-1024, vmax=2048)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.args.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    def trunc(self, mat):
        mat[mat <= self.args.trunc_min] = self.args.trunc_min
        mat[mat >= self.args.trunc_max] = self.args.trunc_max
        return mat

    def trunc_img(self, mat):
        mat[mat < -1024.0] = -1024.0
        mat[mat > 2048.0] = 2048.0
        return mat

    def denormalize_(self, image):
        image = image * (self.args.norm_range_max - self.args.norm_range_min) + self.args.norm_range_min
        return image

    def normalize_(self, image):
        image = (image - self.args.norm_range_min) / (self.args.norm_range_max - self.args.norm_range_min)
        return image

    def restore(self, val_loader, r=None):
        # compute PSNR, SSIM, RMSE, std
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        ori_psnr_std, ori_ssim_std, ori_rmse_std = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        pred_psnr_std, pred_ssim_std, pred_rmse_std = 0, 0, 0
        ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
        pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []
        fanBeam = FanBeam(img_size=768)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for i, (x, y, img_id) in enumerate(val_loader):
                print(f"starting processing from image {img_id}")
                # print(x.shape)
                # print(y.shape)
                # x = x.squeeze(0)
                # y = y.squeeze(0)
                H, W = x.shape[1:]
                # print(x.shape)
                x = self.normalize_(self.trunc(x))
                y = self.normalize_(self.trunc(y))
                x = x.float()
                y = y.float()
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                y = y.flatten(start_dim=0, end_dim=1) if y.ndim == 5 else y
                # print(x.shape)
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
                # break
                x = x.to(self.diffusion.device)
                y = y.to(self.diffusion.device)
                x = data_transform(x)
                # y = data_transform(y)

                pred = self.diffusive_restoration(x)  # .to(self.device)
                pred = inverse_data_transform(pred)
                x = inverse_data_transform(x)

                x = self.trunc(self.denormalize_(x.view(H, W).cpu().data.detach()))
                y = self.trunc(self.denormalize_(y.view(H, W).cpu().data.detach()))
                pred = self.trunc(self.denormalize_(pred.view(H, W).cpu().data.detach()))

                # ####################### "FanBeam-SIRT" #######################################################

                # x_sino = np.float32(x)
                # # x_SIRT = fanBeam.SIRT(VOL=None, proj=x_sino, ang_num=None, iter_num=150)
                # x_SIRT = fanBeam.FBP(x_sino, 720).astype(np.float32)
                # x_SIRT = ((np.float32(x_SIRT) - 0.183) / 0.183 * 1000.).astype(np.float32)
                # # print(np.max(x_SIRT))
                # # # print(x_SIRT.shape)
                # # # print(type(pred_SIRT))
                # # x_SIRT = self.trunc(x_SIRT)
                # x_SIRT = self.trunc_img(torch.from_numpy(x_SIRT))
                #
                # # # x_sino = (np.float32(x) / 1000.0 * 0.183 + 0.183).astype(np.float32)
                # y_sino = np.float32(y)
                # # y_SIRT = fanBeam.SIRT(VOL=None, proj=y_sino, ang_num=None, iter_num=150)
                # y_SIRT = fanBeam.FBP(y_sino, 720).astype(np.float32)
                # y_SIRT = ((np.float32(y_SIRT) - 0.183) / 0.183 * 1000.).astype(np.float32)
                # # print(np.max(y_SIRT))
                # # # print(y_SIRT.shape)
                # # # print(type(pred_SIRT))
                # # y_SIRT = self.trunc(y_SIRT)
                # y_SIRT = self.trunc_img(torch.from_numpy(y_SIRT))
                # # output_file_path = os.path.join(r'D:\wangjiping\UnreDiff-data\stage2\img\test-small-new', f"TEST_{str(i+0).zfill(6)}_target.nii.gz")
                # # sitk.WriteImage(sitk.GetImageFromArray(y_SIRT), output_file_path)

                pred_sino = np.float32(pred)
                pred_SIRT = fanBeam.SIRT(VOL=None, proj=pred_sino, ang_num=None, iter_num=150)
                pred_SIRT = ((np.float32(pred_SIRT) - 0.183) / 0.183 * 1000.).astype(np.float32)
                pred_SIRT = self.trunc_img(torch.from_numpy(pred_SIRT))
                # output_file_path = os.path.join(r'/data/wangjiping/Diffusion-model/hun-data/stage1-new/img/train/input-img-HunreDiff-stage3/', f"l_{str(i*6+0).zfill(6)}_input.nii.gz")
                # sitk.WriteImage(sitk.GetImageFromArray(pred_SIRT), output_file_path)


                # # ####################### "FanBeam-SIRT" #######################################################
                x_sino = np.float32(x)
                x_SIRT = fanBeam.SIRT(VOL=None, proj=x_sino, ang_num=None, iter_num=150)
                x_SIRT = ((np.float32(x_SIRT) - 0.183) / 0.183 * 1000.).astype(np.float32)
                x_SIRT = self.trunc_img(torch.from_numpy(x_SIRT))
                #
                y_sino = np.float32(y)
                y_SIRT = fanBeam.SIRT(VOL=None, proj=y_sino, ang_num=None, iter_num=150)
                y_SIRT = ((np.float32(y_SIRT) - 0.183) / 0.183 * 1000.).astype(np.float32)
                y_SIRT = self.trunc_img(torch.from_numpy(y_SIRT))

                # ####################### "FanBeam-FP" #######################################################
                # pred_img = np.float32(pred) / 1000.0 * 0.183 + 0.183
                # pred_sino = fanBeam.FP(pred_img, 720).astype(np.float32)
                # output_file_path = os.path.join(r'D:\wangjiping\UnreDiff-data\stage1-new\test\input-sino-HunreDiff-stage2-sln', f"n_{str(i*6+0).zfill(6)}_input.nii.gz")  # train
                # output_file_path = os.path.join(r'/data/wangjiping/Diffusion-model/hun-data/stage1-new/img/train/input-sino-HunreDiff-stage3/', f"n_{str(i * 6 + 0).zfill(6)}_input.nii.gz")
                # sitk.WriteImage(sitk.GetImageFromArray(pred_sino), output_file_path)
                # output_file_path_ = os.path.join(r'D:\wangjiping\UnreDiff-data\stage1-new\real-data\fanbeam\input-sino-HunreDiff-stage1-sln-small', f"s_{str(i * 1 + 0).zfill(6)}_target.nii.gz")  # test
                # sitk.WriteImage(sitk.GetImageFromArray(pred_sino), output_file_path_)
                # print("pred_sino.shape = ", pred_sino.shape)

                # y_img = np.float32(y) / 1000.0 * 0.183 + 0.183
                # y_sino = fanBeam.FP(y_img, 720).astype(np.float32)
                # print("y_sino.shape = ", y_sino.shape)
                # # print("min(y_sino) = ", np.min(y_sino))
                # # print("max(y_sino) = ", np.max(y_sino))
                # with open(r'D:\wangjiping\UnreDiff-data\stage1-new\train/y_value.txt', 'a') as f:
                #     f.write('MIN:%.10f' % (np.min(y_sino)) + '\n')
                #     f.write('MAX:%.10f' % (np.max(y_sino)) + '\n')
                #     f.close()
                # output_file_path = os.path.join(r'F:\CT\Train\data-new\stage1-new\img\test\target-sino', f"TEST_{str(i*1).zfill(6)}_target.nii.gz")
                # sitk.WriteImage(sitk.GetImageFromArray(y_sino), output_file_path)

                # y_img1 = fanBeam.FBP(y_sino, 720).astype(np.float32)
                # # fv_img[fv_img < 0.0] = 0.0
                # y_img1 = (y_img1 - 0.183) / 0.183 * 1000.
                # # fv_img_s = sitk.GetImageFromArray(np.float32(fv_img))
                # output_file_path = os.path.join(r'D:\wangjiping\UnreDiff-data\stage1\sino\test-small', f"lT_{str(i).zfill(6)}_img.nii.gz")
                # sitk.WriteImage(sitk.GetImageFromArray(np.float32(y_img1)), output_file_path)
                #####################################################################################

                output_file_path = os.path.join(self.args.save_path, 'x', f"l_{str(i).zfill(6)}_input.nii.gz")
                sitk.WriteImage(sitk.GetImageFromArray(x_SIRT), output_file_path)
                output_file_path = os.path.join(self.args.save_path, 'y', f"{str(i).zfill(6)}_target.nii.gz")
                sitk.WriteImage(sitk.GetImageFromArray(y_SIRT), output_file_path)
                output_file_path = os.path.join(self.args.save_path, 'pred', f"{str(i).zfill(6)}_pred.nii.gz")
                sitk.WriteImage(sitk.GetImageFromArray(pred_SIRT), output_file_path)

                # np.save(os.path.join(self.args.save_path, 'x', '{}_result'.format(i)), x)
                # np.save(os.path.join(self.args.save_path, 'y', '{}_result'.format(i)), y)
                # np.save(os.path.join(self.args.save_path, 'pred', '{}_result'.format(i)), pred)

                data_range = 3072

                original_result, pred_result = compute_measure(x_SIRT, y_SIRT, pred_SIRT, data_range)
                ori_psnr_avg += original_result[0]
                ori_psnr_avg1.append(original_result[0])
                ori_ssim_avg += original_result[1]
                ori_ssim_avg1.append(original_result[1])
                ori_rmse_avg += original_result[2]
                ori_rmse_avg1.append(original_result[2])
                pred_psnr_avg += pred_result[0]
                pred_psnr_avg1.append(pred_result[0])
                pred_ssim_avg += pred_result[1]
                pred_ssim_avg1.append(pred_result[1])
                pred_rmse_avg += pred_result[2]
                pred_rmse_avg1.append(pred_result[2])

                # save result figure
                if self.args.result_fig:
                    self.save_fig(x_SIRT, y_SIRT, pred_SIRT, i, original_result, pred_result)

                # testtime
            torch.cuda.synchronize()
            end = time.time()

            # calculate STD
            for i in range(len(val_loader)):
                ori_psnr_std += (ori_psnr_avg1[i] - ori_psnr_avg / len(val_loader)) ** 2
                ori_ssim_std += (ori_ssim_avg1[i] - ori_ssim_avg / len(val_loader)) ** 2
                ori_rmse_std += (ori_rmse_avg1[i] - ori_rmse_avg / len(val_loader)) ** 2

                pred_psnr_std += (pred_psnr_avg1[i] - pred_psnr_avg / len(val_loader)) ** 2
                pred_ssim_std += (pred_ssim_avg1[i] - pred_ssim_avg / len(val_loader)) ** 2
                pred_rmse_std += (pred_rmse_avg1[i] - pred_rmse_avg / len(val_loader)) ** 2

            # # testtime
            # torch.cuda.synchronize()
            # end = time.time()

            print('\n')
            print(
                'Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nPSNR std: {:.4f} \nSSIM std: {:.4f} \nRMSE std: {:.4f}'.format(
                    ori_psnr_avg / len(val_loader), ori_ssim_avg / len(val_loader),
                    ori_rmse_avg / len(val_loader),
                    pow(ori_psnr_std / len(val_loader), 0.5), pow(ori_ssim_std / len(val_loader), 0.5),
                    pow(ori_rmse_std / len(val_loader), 0.5)))
            print(
                'After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nPSNR std: {:.4f} \nSSIM std: {:.4f} \nRMSE std: {:.4f}'.format(
                    pred_psnr_avg / len(val_loader), pred_ssim_avg / len(val_loader),
                    pred_rmse_avg / len(val_loader),
                    pow(pred_psnr_std / len(val_loader), 0.5), pow(pred_ssim_std / len(val_loader), 0.5),
                    pow(pred_rmse_std / len(val_loader), 0.5)))
            print('\n')
            print('Test time: {:.4f} s'.format(end - start))
            # #
            # # # utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}_output.png"))
            # #
            # #
