import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
from datasets.data_augment import augment_img
import random
from torch.utils.data import Dataset, DataLoader
from glob import glob
import re
import SimpleITK as sitk
import zipfile


class CT:
    def __init__(self, config):
        self.config = config

    def get_loaders(self, parse_patches=False):
        print("=> evaluating CT test set...")
        train_dataset = ct_dataset(mode='train',
                                   load_mode=0,
                                   augment=self.config.data.augment,
                                   saved_path=os.path.join(self.config.data.data_dir, 'train'),
                                   test_patient='val',
                                   transforms=None,
                                   )
        val_dataset = ct_dataset(mode='test',
                                 load_mode=0,
                                 augment=False,
                                 saved_path=os.path.join(self.config.data.data_dir, 'test'),
                                 test_patient='TEST_',
                                 transforms=None,
                                 )

        # if not parse_patches:
        #     self.config.training.batch_size = 1
        #     self.config.sampling.batch_size = 1

        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)
        print('len(train_loader)=', len(train_loader))
        print('len(val_loader)=', len(val_loader))

        return train_loader, val_loader


class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, augment, saved_path, test_patient,  transforms=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0, 1], "load_mode is 0 or 1"
        assert augment in [True, False], "augment True or False"

        input_path = sorted(glob(os.path.join(saved_path, 'input-sino-HunreDiff-stage2', '*_input.nii.gz')))  # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        target_path = sorted(glob(os.path.join(saved_path, 'target-sino', '*_target.nii.gz')))

        # input_path = sorted(glob(os.path.join(saved_path, '*_input.nii.gz')))  # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        # target_path = sorted(glob(os.path.join(saved_path, '*_target.nii.gz')))

        self.load_mode = load_mode
        self.transforms = transforms
        self.augment = augment

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:  # all data load
                self.input_ = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in input_]
                self.target_ = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in target_]
                # self.input_ = [np.load(f) for f in input_]
                # self.target_ = [np.load(f) for f in target_]
        else:  # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:    # all data load
                self.input_ = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in input_]
                self.target_ = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in target_]
                # self.input_ = [np.load(f) for f in input_]
                # self.target_ = [np.load(f) for f in target_]

    @staticmethod
    def get_patch(full_input_img, full_target_img, patch_n, patch_size):  # 定义patch
        assert full_input_img.shape == full_target_img.shape
        patch_input_imgs = []
        patch_target_imgs = []
        h, w = full_input_img.shape
        new_h, new_w = patch_size, patch_size
        for _ in range(patch_n):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch_input_img = full_input_img[top:top + new_h, left:left + new_w]
            patch_target_img = full_target_img[top:top + new_h, left:left + new_w]
            patch_input_imgs.append(patch_input_img)
            patch_target_imgs.append(patch_target_img)
        # return np.array(patch_input_imgs), np.array(patch_target_imgs)
        return tuple(patch_input_imgs), tuple(patch_target_imgs)

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        # print(input_img)
        img_id = re.split('/', input_img)[-1][:-7]
        if self.load_mode == 0:
            input_img, target_img = sitk.GetArrayFromImage(sitk.ReadImage(input_img)), sitk.GetArrayFromImage(sitk.ReadImage(target_img))

        # if self.augment:
        #     temp = np.random.randint(0, 8)
        #     input_img, target_img = augment_img(input_img, temp), augment_img(target_img, temp)

        if self.transforms:
            input_img = self.transforms(input_img)
            target_img = self.transforms(target_img)

        # input_img, target_img = torch.tensor(input_img).unsqueeze(0), torch.tensor(target_img).unsqueeze(0)

        # print(input_img.shape)

        return input_img, target_img, img_id


