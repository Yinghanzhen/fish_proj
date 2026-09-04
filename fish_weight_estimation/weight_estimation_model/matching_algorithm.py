import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from fish_weight_estimation.fish_class import TopFishData, SideFishData


def compute_straightness_factor(pts_2d: List[List[float]]) -> float:
    """安全计算俯视骨架线的姿态伸展因子"""
    if not pts_2d or len(pts_2d) < 2:
        return 1.0

    pts = np.array(pts_2d, dtype=np.float32)
    d_straight = np.linalg.norm(pts[0] - pts[-1])  # 头尾直线距离
    l_spine = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))  # 沿骨架弧长

    if l_spine <= 1e-5:
        return 1.0

    return float(np.clip(d_straight / l_spine, 0.05, 1.0))


def cross_view_fish_matching(
        top_fishes: List[TopFishData],
        side_fishes: List[SideFishData],
        w_x: float = 0.5,
        w_depth: float = 0.3,
        w_pose: float = 0.2
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    基于俯视/侧视鱼实例列表的双视角匈牙利匹配算法 (修复版)
    """
    N, M = len(top_fishes), len(side_fishes)
    if N == 0 or M == 0:
        return [], np.empty((N, M))

    # 1. 提取归一化的 X 排序（包含容差机制）
    top_x = np.array([f.bbox[0] for f in top_fishes])
    side_x = np.array([f.bbox[0] for f in side_fishes])

    def compute_relative_x_rank(x_arr):
        ranks = np.zeros(len(x_arr))
        for i in range(len(x_arr)):
            # 核心改进：只要 X 坐标相差 10 个像素以内，我们就认为它们处于同一个“水平段”，不扣分
            # 10是一个经验阈值，你可以根据你的图像分辨率微调 (如 300x400 的图像设为 15)
            x_tolerance = 10.0
            smaller = np.sum(x_arr < (x_arr[i] - x_tolerance))
            equal = np.sum(np.abs(x_arr - x_arr[i]) <= x_tolerance)
            # 排名 = 严格小于它的个数 / 总数 (引入容差后，相同的 X 得到相同的 Rank)
            ranks[i] = smaller / max(1, len(x_arr) - 1)
        return ranks

    top_x_ranks = compute_relative_x_rank(top_x)
    side_x_ranks = compute_relative_x_rank(side_x)

    # 2. 提取纵深相对位置 (【修复核心】纠正映射方向)
    top_y = np.array([f.y_mid for f in top_fishes])
    side_depths = np.array([f.top_kpt_3d[2] if f.top_kpt_3d else 0.0 for f in side_fishes])

    def compute_relative_depth_rank(y_arr, depth_arr):
        # 对俯视图 Y: 值越小越在上方(远)。我们将其归一化为 0~1 (0=最远, 1=最近)
        # 对侧视图深: 值越大越远。我们将其归一化为 0~1 (0=最近, 1=最远)
        y_norm = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min() + 1e-6)
        depth_norm = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-6)

        # 映射关系必须统一！
        # 俯视图中 y 越小（靠上） <=> 侧视图中 depth 越大（靠后）
        # 所以我们计算代价时，直接用它们归一化后的差值即可（因为都变成了 0~1 同序）
        return y_norm, depth_norm

    top_y_ranks, side_depth_ranks = compute_relative_depth_rank(top_y, side_depths)

    # 3. 构建匈牙利代价矩阵 Cost Matrix
    cost_matrix = np.zeros((N, M), dtype=np.float32)

    for i, t_fish in enumerate(top_fishes):
        gamma = compute_straightness_factor(t_fish.spine_2d)
        effective_length = (t_fish.spine_length or 0.0) * gamma

        for j, s_fish in enumerate(side_fishes):
            # 约束 1: X 轴位置 Cost
            cost_x = abs(top_x_ranks[i] - side_x_ranks[j])

            # 约束 2: 纵深排序 Cost (修改后的映射)
            cost_depth_y = abs(top_y_ranks[i] - side_depth_ranks[j])

            # 约束 3: 体高/体长 (H/L) 比例 Consistency Cost
            h_side = s_fish.body_height_3d or 0.0
            hl_ratio = h_side / (effective_length + 1e-5)
            cost_pose = abs(hl_ratio - 0.27)

            # 加权求和
            cost_matrix[i, j] = (w_x * cost_x) + (w_depth * cost_depth_y) + (w_pose * cost_pose)

    # 4. 匈牙利算法求解全局最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = list(zip(row_ind, col_ind))

    return matches, cost_matrix