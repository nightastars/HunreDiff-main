import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.vit_inr import Transformer_INR
from utils.measure import compute_measure
import utils.weight_init as w_init
from torch.optim import lr_scheduler
import imageio
# from torch.cuda.amp import GradScaler
from utils.indi_diffusion import Indi_cond
import matplotlib.pyplot as plt
# gscaler = GradScaler()


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.diffusion = Indi_cond()
        self.model = Transformer_INR(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.mseloss = nn.MSELoss(reduction='sum')
        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, config.training.n_epochs, eta_min=1e-6)
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        # checkpoint = utils.logging.load_checkpoint(os.path.join(load_path, 'ckpts', 'CT_73_ddpm.pth.tar'), None)
        checkpoint = utils.logging.load_checkpoint(os.path.join(load_path), None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.optimizer.state_dict()['param_groups'][0]['lr'] = 3e-5
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def normalize_(self, image):
        image = (image - self.args.norm_range_min) / (self.args.norm_range_max - self.args.norm_range_min)
        return image

    def denormalize_(self, image):
        image = image * (self.args.norm_range_max - self.args.norm_range_min) + self.args.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat < self.args.trunc_min] = self.args.trunc_min
        mat[mat > self.args.trunc_max] = self.args.trunc_max
        return mat

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders(parse_patches=self.args.parse_patches)
        params = self.get_parameter_number(self.model)
        print(params)
        # print("------init_model------")
        # w_init.init_weights(self.model)
        #if os.path.isfile(self.args.save_path):
        self.load_ddm_ckpt(self.args.save_path)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y, _) in enumerate(train_loader):
                x = self.normalize_(self.trunc(x))
                y = self.normalize_(self.trunc(y))
                # print(x.shape)
                # print(x)
                x = x.float()
                y = y.float()
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                y = y.flatten(start_dim=0, end_dim=1) if y.ndim == 5 else y
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
                # print(x.shape)

                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                y = y.to(self.device)
                x = data_transform(x)
                y = data_transform(y)

                # print(x.shape)
                # print(y.shape)

                t = torch.rand(size=(y.shape[0],)).to(self.device)

                fct = t[:, None, None, None]
                transformed_image = (1 - fct) * y + fct * x
                # print(transformed_image.shape)
                predicted_peak = self.model(transformed_image, t)

                loss = self.mseloss(y, predicted_peak)

                if self.step % 100 == 0:
                    print(f"step: {self.step}, epoch: {epoch}, lr: {self.optimizer.state_dict()['param_groups'][0]['lr']},"
                          f" loss: {loss.item()}, data time: {data_time / (i+1)}")

                # gscaler.scale(loss).backward()
                # gscaler.step(self.optimizer)
                # gscaler.update()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.args.save_path, 'ckpts', self.config.data.dataset + '_' + str(epoch+1) + '_ddpm'))

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step, len(train_loader))

            # self.scheduler.step()

    def sample_image(self, x_cond):
        xs = self.diffusion.indisample(x_cond, self.model, self.args.sampling_timesteps)
        return xs

    def save_fig(self, x, y, pred, fig_name, file_directory):
        x, y, pred = x.cpu().numpy(), y.cpu().numpy(), pred.cpu().numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.args.trunc_min, vmax=self.args.trunc_max)
        ax[0].set_title('SVCT', fontsize=30)
        # ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
        #                                                                    original_result[1],
        #                                                                    original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.args.trunc_min, vmax=self.args.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        # ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
        #                                                                    pred_result[1],
        #                                                                    pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.args.trunc_min, vmax=self.args.trunc_max)
        ax[2].set_title('FVCT', fontsize=30)

        f.savefig(os.path.join(file_directory, 'result_{}.png'.format(fig_name)))
        plt.close()

    def sample_validation_patches(self, val_loader, step, train_len):
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0., 0., 0.
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y, _) in enumerate(val_loader):
                H, W = x.shape[1:]
                # print(x.shape)
                x = self.normalize_(self.trunc(x))
                y = self.normalize_(self.trunc(y))
                x = x.float()
                y = y.float()
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                y = y.flatten(start_dim=0, end_dim=1) if y.ndim == 5 else y
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
                # break
                x = x.to(self.device)
                y = y.to(self.device)
                x = data_transform(x)
                # y = data_transform(y)

                pred = self.sample_image(x)   # .to(self.device)
                pred = inverse_data_transform(pred)
                x = inverse_data_transform(x)
                # y = inverse_data_transform(y)

                x = self.trunc(self.denormalize_(x))
                gt = self.trunc(self.denormalize_(y))
                pred = self.trunc(self.denormalize_(pred))
                data_range = self.args.trunc_max - self.args.trunc_min
                # data_range = torch.tensor(data_range, device=self.device)
                # print('data_range.device()=', data_range.device)
                # print('x.device()=', x.device)
                # print('gt.device()=', gt.device)
                # print('pred.device()=', pred.device)
                original_result, pred_result = compute_measure(x, gt, pred, data_range)

                #         pred_niqe_avg += self.niqe_metric(target_pred)
                #
                # with open(self.save_path + '/pred_niqe_avg.txt', 'a') as f:
                #     f.write('EPOCH:%d loss:%.20f' % (epoch, pred_niqe_avg / len(self.val_data_loader)) + '\n')
                #     f.close()

                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                file_directory = os.path.join(self.args.images_path, str(step))
                if not os.path.exists(file_directory):
                    os.makedirs(file_directory)
                    print('Create path : {}'.format(file_directory))

                # save result figure
                self.save_fig(x.view(H, W), gt.view(H, W), pred.view(H, W), i, file_directory)
                
                # imageio.imwrite((os.path.join(file_directory, '{}_cond.png'.format(i))), x.view(self.config.data.image_size, self.config.data.image_size).to(torch.uint8).cpu())
                # imageio.imwrite((os.path.join(file_directory, '{}_pred.png'.format(i))), pred.view(self.config.data.image_size, self.config.data.image_size).to(torch.uint8).cpu())
                # imageio.imwrite((os.path.join(file_directory, '{}_label.png'.format(i))), gt.view(self.config.data.image_size, self.config.data.image_size).to(torch.uint8).cpu())
                # utils.logging.save_image(x, os.path.join(self.args.images_path, str(step), f"{i}_cond.png"))

            # 日志文件
            # with open(self.save_path+'/disc_loss.txt', 'a') as f:
            #     f.write('EPOCH:%d loss:%.20f' % (disc_loss) + '\n')
            #     f.close()

            with open(self.args.save_path + '/pred_psnr_avg.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (step / train_len, pred_psnr_avg / len(val_loader)) + '\n')
                f.close()

            with open(self.args.save_path + '/pred_ssim_avg.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (step / train_len, pred_ssim_avg / len(val_loader)) + '\n')
                f.close()

            with open(self.args.save_path + '/pred_rmse_avg.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (step / train_len, pred_rmse_avg / len(val_loader)) + '\n')
                f.close()

            # for i in range(n):
            #     utils.logging.save_image(x_cond[i], os.path.join(self.args.images_path, str(step), f"{i}_cond.png"))
            #     utils.logging.save_image(x[i], os.path.join(self.args.images_path, str(step), f"{i}.png"))
