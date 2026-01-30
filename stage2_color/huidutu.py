"""
Author:wanyii
time:2023/10
"""

import os
import cv2
from PIL import Image
import numpy as np


def convert_to_grayscale_opencv(input_dir, output_dir):
    """
    使用OpenCV将彩色图像转换为灰度图像
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            # 构建完整的文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # 读取彩色图像
                color_image = cv2.imread(input_path)

                if color_image is not None:
                    # 转换为灰度图像
                    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                    # 保存灰度图像
                    cv2.imwrite(output_path, gray_image)
                    print(f"已转换: {filename}")
                else:
                    print(f"无法读取文件: {filename}")

            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")


def convert_to_grayscale_pil(input_dir, output_dir):
    """
    使用PIL将彩色图像转换为灰度图像
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            # 构建完整的文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # 打开彩色图像
                with Image.open(input_path) as color_image:
                    # 转换为灰度图像
                    gray_image = color_image.convert('L')

                    # 保存灰度图像
                    gray_image.save(output_path)
                    print(f"已转换: {filename}")

            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")


def main():
    # 设置输入和输出目录路径
    input_directory = "assets/fengjing"
    output_directory = "Y:/teacherguo/DDColor-master_create01/assets/fengjing_grey1"

    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"输入目录不存在: {input_directory}")
        return

    print("开始转换图像...")
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory}")
    print("-" * 50)

    # 方法1: 使用OpenCV (推荐，处理速度较快)
    print("使用OpenCV方法转换:")
    convert_to_grayscale_opencv(input_directory, output_directory)

    # 方法2: 使用PIL (备选方案)
    # print("使用PIL方法转换:")
    # convert_to_grayscale_pil(input_directory, output_directory)

    print("-" * 50)
    print("转换完成!")


if __name__ == "__main__":
    main()