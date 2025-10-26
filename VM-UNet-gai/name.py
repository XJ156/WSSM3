
import os

def rename_files(folder_path, old_name, new_name):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            new_file_path = os.path.join(root, file.replace(old_name, new_name))
            os.rename(file_path, new_file_path)

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            new_dir_path = os.path.join(root, dir.replace(old_name, new_name))
            os.rename(dir_path, new_dir_path)

# 使用示例
folder_path = "/disk/sdb/zhangqian0620/VM-UNet-main/isic_256"
old_name = "_segmentation"
new_name = ""

rename_files(folder_path, old_name, new_name)
