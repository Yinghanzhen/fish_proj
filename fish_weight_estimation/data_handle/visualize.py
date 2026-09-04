import cv2
import numpy as np
from typing import List, Union
from pathlib import Path


def visualize_fishes_weight(
        img_input: Union[str, Path, np.ndarray],
        top_fishes: List,
        output_path: str = "top_view_weight_result.jpg",
        box_color: tuple = (0, 255, 0),
        txt_color: tuple = (0, 0, 0)
) -> np.ndarray:
    """
    在俯视图上绘制鱼的边界框并标注体重信息
    :param img_input: 图像路径 (str/Path) 或已读取的 OpenCV 图像阵列 (np.ndarray)
    :param top_fishes: 包含 TopFishData 实例的列表
    :param output_path: 保存渲染图的目标路径
    :param box_color: 边界框颜色，默认为绿色 (BGR: 0, 255, 0)
    :param txt_color: 文字颜色，默认为黑色 (BGR: 0, 0, 0)
    :return: 绘制完毕的 OpenCV 图像矩阵 (np.ndarray)
    """
    # 1. 兼容输入类型：路径或 Mat 矩阵
    if isinstance(img_input, (str, Path)):
        img = cv2.imread(str(img_input))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_input}")
    else:
        img = img_input.copy()

    # 2. 循环绘制边界框与体重
    for fish in top_fishes:
        if not hasattr(fish, 'bbox') or fish.bbox is None:
            continue

        # 转换为整数像素坐标 [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = map(int, fish.bbox)

        # 绘制主边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

        # 提取体重与 ID 信息
        weight_val = getattr(fish, "weight", 0.0)
        label = f"ID:{fish.fish_id} {weight_val:.2f}g"

        # 计算文字背景框维度（防止靠边遮挡）
        (font_w, font_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_y = max(y1 - 10, font_h + 10)

        # 绘制文本背景块与文字
        cv2.rectangle(
            img,
            (x1, text_y - font_h - 4),
            (x1 + font_w + 4, text_y + baseline - 2),
            box_color,
            -1
        )
        cv2.putText(
            img,
            label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            txt_color,
            2,
            cv2.LINE_AA
        )

    # 3. 自动保存结果
    if output_path:
        cv2.imwrite(output_path, img)

    return img