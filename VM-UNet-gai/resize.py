from PIL import Image
import os


# 原始文件夹路径
original_folder = '/disk/sdb/zhangqian0620/VM-UNet-main/jieguo/output_olp_box'
# 新文件夹路径
new_folder = '/disk/sdb/zhangqian0620/VM-UNet-main/jieguo/olp_pre_224'

# 如果新文件夹不存在，则创建新文件夹
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 遍历原始文件夹中的所有文件
for filename in os.listdir(original_folder):
    if filename.endswith('.png'):
        # 读取原始图片
        image_path = os.path.join(original_folder, filename)
        image = Image.open(image_path)

        # 调整图片大小为128*128
        resized_image = image.resize((224, 224))

        # 生成新文件路径
        new_image_path = os.path.join(new_folder, filename)

        # 保存调整后的图片
        resized_image.save(new_image_path)


