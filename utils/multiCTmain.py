import sys
import os
import numpy as np
from numpy import matlib
# from .loadData import loadData
from scipy.interpolate import griddata
from scipy.signal import medfilt2d
import astra
import scipy.io as sio
import SimpleITK as sitk


class FanBeam():
    def __init__(self, img_size):
        self.limited_projGeom720 = astra.create_proj_geom('fanflat', 2.0, 740, np.linspace(0, 2 * np.pi * 1, 720, endpoint=False), 1270, 870)  # 1.8, 1000
        # self.limited_projGeom720 = astra.create_proj_geom('fanflat', 2.0, 912, np.linspace(0, 2 * np.pi * 1, 720, endpoint=False), 1085.6, 595.0)  # 1.8, 1000
        # self.limited_projGeom180 = astra.create_proj_geom('fanflat', 2.0, 768, np.linspace(0, np.pi/2, 180, endpoint=False), 512, 512)#1.8, 1000
        # self.limited_projGeom60 = astra.create_proj_geom('fanflat', 2.0, 768, np.linspace(0, np.pi/3, 120, endpoint=False), 512, 512)
        self.volGeom = astra.create_vol_geom(img_size, img_size, (-img_size / 2) * 500 / img_size, (img_size / 2) * 500 / img_size, (-img_size / 2) * 500 / img_size, (img_size / 2) * 500 / img_size)
        # self.volGeom = astra.create_vol_geom(img_size, img_size, -1 * img_size / 4, img_size / 4, -1 * img_size / 4, img_size / 4)

    def FP(self, img, ang_num):
        # if ang_num == 90:
        #     projGeom = self.limited_projGeom180
        # elif ang_num == 60:
        #     projGeom = self.limited_projGeom60
        projGeom = self.limited_projGeom720

        volGeom = self.volGeom
        rec_id = astra.data2d.create('-vol', volGeom, img)
        proj_id = astra.data2d.create('-sino', projGeom)
        cfg = astra.astra_dict('FP_CUDA')
        cfg['VolumeDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id).T
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return pro

    def FBP(self, proj, ang_num):
        # if ang_num == 90:
        #     projGeom = self.limited_projGeom180
        # elif ang_num == 60:
        #     projGeom = self.limited_projGeom60

        projGeom = self.limited_projGeom720

        volGeom = self.volGeom
        rec_id = astra.data2d.create('-vol', volGeom)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def SIRT(self, VOL, proj, ang_num, iter_num):
        # if ang_num == 90:
        #     projGeom = self.limited_projGeom180
        # elif ang_num == 60:
        #     projGeom = self.limited_projGeom60

        projGeom = self.limited_projGeom720
        volGeom = self.volGeom
        if VOL is None:
            rec_id = astra.data2d.create('-vol', volGeom)
        else:
            rec_id = astra.data2d.create('-vol', volGeom, VOL)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iter_num)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec


if __name__ == '__main__':
    fanBeam = FanBeam(img_size=512)
    def trunc(mat, trunc_min, trunc_max):
        mat[mat <= trunc_min] = trunc_min
        mat[mat >= trunc_max] = trunc_max
        return mat
    # 设置输入和输出文件夹路径
    input_folder = r'C:\Users\jipin\Desktop\nii-test'
    output_folder = r'C:\Users\jipin\Desktop\result'
    print("len(input_folder)=", len(input_folder))
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    k = 0
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            # 构建完整的文件路径
            file_path = os.path.join(input_folder, filename)
            print(file_path)

            # 使用SimpleITK读取nii文件
            image = sitk.ReadImage(file_path)

            # 将SimpleITK图像转换为numpy数组
            image_array = sitk.GetArrayFromImage(image)
            image_array = trunc(image_array, -1024, 2048)
            # image_array = image_array.reshape(image_array.shape[1:])
            image_array = np.float32(image_array) / 1000.0 * 0.183 + 0.183
            fv_sino = fanBeam.FP(image_array, 720).astype(np.float32)
            fv_sino_s = sitk.GetImageFromArray(np.float32(fv_sino))
            output_file_path = os.path.join(output_folder, f"TEST_{str(k).zfill(6)}_sino.nii.gz")
            sitk.WriteImage(fv_sino_s, output_file_path)

            fv_img = fanBeam.FBP(fv_sino, 720).astype(np.float32)
            # fv_img[fv_img < 0.0] = 0.0
            fv_img = (fv_img - 0.183) / 0.183 * 1000.
            fv_img_s = sitk.GetImageFromArray(np.float32(fv_img))
            output_file_path = os.path.join(output_folder, f"TEST_{str(k).zfill(6)}_img.nii.gz")
            sitk.WriteImage(fv_img_s, output_file_path)

            # fv_img, sv_image_array = (fv_img - 0.183) / 0.183 * 1000., (sv_image_array - 0.183) / 0.183 * 1000.
            #
            # fv_img, sv_image_array = sitk.GetImageFromArray(np.float32(fv_img)), sitk.GetImageFromArray(
            #     np.float32(sv_image_array))
            #
            # # 构建输出文件路径
            # output_file_path = os.path.join(output_folder, f"TEST_{str(k).zfill(6)}_input.nii.gz")
            # sitk.WriteImage(sv_image_array, output_file_path)
            # # 构建输出文件路径
            # output_file_path = os.path.join(output_folder, f"TEST_{str(k).zfill(6)}_target.nii.gz")
            # sitk.WriteImage(fv_img, output_file_path)

            k += 1