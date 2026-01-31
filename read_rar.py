import os
import rarfile  # 需要安装 rarfile 库

# 假设 saved_path 是一个 RAR 文件路径
saved_path = r'/data/wangjiping/Diffusion-model/hun-data/stage1-new/img/train/target-img.rar'  # 替换为你的 RAR 文件路径
extract_to = r'/data/wangjiping/Diffusion-model/hun-data/stage1-new/img/train/'  # 临时解压目录

# 创建解压目录（如果不存在）
os.makedirs(extract_to, exist_ok=True)

# 解压 RAR 文件
with rarfile.RarFile(saved_path) as rf:
    rf.extractall(extract_to)
