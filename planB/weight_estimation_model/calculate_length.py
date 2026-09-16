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
        diffs = np.diff(pts, axis=0)#相邻元素按列求差
        segment_lengths = np.linalg.norm(diffs, axis=1)#求每段之间长度
        fish.spine_length = float(np.sum(segment_lengths))