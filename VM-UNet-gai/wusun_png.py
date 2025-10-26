import cv2
import os

# 定义源图片和目标图片的文件夹路径
src_folder = "/disk/sdb/zhangqian0620/VM-UNet-main/output_isic"
dst_folder = "/disk/sdb/zhangqian0620/VM-UNet-main/output_isic_png"

# 循环遍历源文件夹内的所有文件
for filename in os.listdir(src_folder):
    # 获取文件的绝对路径
    src_path = os.path.join(src_folder, filename)

    # 只处理JPG格式的文件，忽略其他格式的文件
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # 读取源文件的图像数据
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)

        # 将图像数据转换为PNG格式，压缩质量为0（最高质量）
        dst_path = os.path.join(dst_folder, os.path.splitext(filename)[0] + ".png")
        cv2.imwrite(dst_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])