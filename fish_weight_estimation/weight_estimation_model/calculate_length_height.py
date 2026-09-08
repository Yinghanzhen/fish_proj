import numpy as np
from typing import List
from fish_weight_estimation.fish_camera_class import TopFishData, SideFishData

def calculate_lengths(top_fishes: List[TopFishData]) -> None:
    """
    计算俯视鱼列表中每条鱼 3D 骨架线的累加弧长（体长），并自动更新实例的 spine_length 属性。

    参数:
        top_fishes: 该帧所有 TopFishData 实例构成的列表
    """
    for fish in top_fishes:
        # 如果没有 3D 骨架或点数少于 2，体长直接设为 0.0
        if fish.spine_3d is None or len(fish.spine_3d) < 2:
            fish.spine_length = 0.0
            continue

        pts = np.asarray(fish.spine_3d, dtype=np.float64)

        # 计算相邻点之间的 3D 欧氏距离并求和
        diffs = np.diff(pts, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        fish.spine_length = float(np.sum(segment_lengths))


def calculate_heights(side_fishes: List[SideFishData], pixel_to_mm_scale: float = 1.0) -> None:
    """
    计算侧视鱼列表中每条鱼背腹两点间的体高，并自动更新实例的 body_height_3d 属性。
    参数:
        side_fishes: 该帧所有 SideFishData 实例构成的列表
        pixel_to_mm_scale: 2D 回退时的像素到物理尺寸比例（可选）
    """
    for fish in side_fishes:
        # 1. 优先使用 3D 坐标计算（真实物理尺寸）
        if fish.top_kpt_3d is not None and fish.bottom_kpt_3d is not None:
            top_pt = np.asarray(fish.top_kpt_3d, dtype=np.float64)
            bottom_pt = np.asarray(fish.bottom_kpt_3d, dtype=np.float64)
            fish.body_height_3d = float(np.linalg.norm(top_pt - bottom_pt))

        # 2. 回退使用 2D 像素坐标计算
        elif fish.top_kpt_2d is not None and fish.bottom_kpt_2d is not None:
            top_pt = np.asarray(fish.top_kpt_2d, dtype=np.float64)
            bottom_pt = np.asarray(fish.bottom_kpt_2d, dtype=np.float64)
            pixel_height = float(np.linalg.norm(top_pt - bottom_pt))
            # 乘以比例换算为物理尺寸（若无标定比例则默认 1.0）
            fish.body_height_3d = pixel_height * pixel_to_mm_scale

        else:
            fish.body_height_3d = 0.0