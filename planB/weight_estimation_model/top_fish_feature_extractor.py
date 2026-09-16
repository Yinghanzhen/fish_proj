"""
计算特征包括：
1. 真实 3D 沿线展平体长 (spine_length)
2. 结合 3D 深度与 2D Mask 沿骨架法线切片的物理体宽 (max_body_width, mean_body_width)
3. 2D 弧长展平 + 1D 去趋势解耦后的背部曲率 (back_curvature)
4. 姿态解耦后的净背部最大厚度 (back_height)
"""

from typing import List, Tuple
import cv2
import numpy as np


def extract_body_width_from_3d_spine(fish_data: "TopFishData") -> Tuple[float, float]:
    """
    利用已有的 3D 骨架线 spine_3d 与 2D Mask，沿着骨架法线方向计算真实的物理体宽。
    自动根据每个骨架节点处的真实深度 Z 动态计算像素到物理尺寸的转换因子。

    :param fish_data: 已填充 spine_3d 和 mask 的 TopFishData 对象
    :return: (max_body_width, mean_body_width) 物理尺寸
    """
    if fish_data.mask is None or fish_data.spine_2d is None or fish_data.spine_3d is None:
        return 0.0, 0.0

    spine_2d = np.array(fish_data.spine_2d, dtype=np.float32)
    spine_3d = fish_data.spine_3d  # 形状为 (N, 3)

    # 滤除包含 NaN 的无效点
    valid_mask = ~np.isnan(spine_2d).any(axis=1)
    valid_pts_2d = spine_2d[valid_mask]
    valid_pts_3d = spine_3d[valid_mask]

    num_pts = len(valid_pts_2d)
    if num_pts < 3:
        return 0.0, 0.0

    mask_uint8 = fish_data.mask.astype(np.uint8) * 255
    widths_physical = []

    # 采样 2D 像素与 3D 对应的比例：利用 3D 点间的物理距离除以 2D 点间的像素距离
    diffs_2d = np.linalg.norm(np.diff(valid_pts_2d, axis=0), axis=1)
    diffs_3d_xy = np.linalg.norm(np.diff(valid_pts_3d[:, :2], axis=0), axis=1)

    # 防止除以 0，计算各段的 像素->物理 转换系数 (cm/pixel)
    valid_ratios = np.where(diffs_2d > 1e-3, diffs_3d_xy / diffs_2d, 0.0)

    for i in range(1, num_pts - 1):
        pt_2d = valid_pts_2d[i]
        prev_pt_2d = valid_pts_2d[i - 1]
        next_pt_2d = valid_pts_2d[i + 1]

        # 1. 2D 平面切线与法线向量
        tangent = next_pt_2d - prev_pt_2d
        norm = np.linalg.norm(tangent)
        if norm < 1e-5:
            continue

        normal = np.array([-tangent[1], tangent[0]]) / norm  # 垂直于切线的法线方向

        # 2. 沿法线正反方向射线搜索 Mask 边界像素交点
        line_len = max(mask_uint8.shape) // 2
        p1 = (pt_2d + normal * line_len).astype(np.int32)
        p2 = (pt_2d - normal * line_len).astype(np.int32)

        line_mask = np.zeros_like(mask_uint8)
        cv2.line(line_mask, (p1[0], p1[1]), (p2[0], p2[1]), color=255, thickness=1)
        intersection = cv2.bitwise_and(mask_uint8, line_mask)

        # 3. 统计该切线处的像素宽度
        width_pixel = np.sum(intersection > 0)

        if width_pixel > 0:
            # 获取该节点局部的 像素->物理尺寸 比例系数
            local_scale = valid_ratios[i - 1] if valid_ratios[i - 1] > 0 else np.mean(
                valid_ratios[valid_ratios > 0]) if np.any(valid_ratios > 0) else 1.0

            # 将像素体宽转换为物理体宽
            widths_physical.append(width_pixel * local_scale)

    if not widths_physical:
        return 0.0, 0.0

    max_width = float(np.max(widths_physical))
    mean_width = float(np.mean(widths_physical))

    fish_data.max_body_width = max_width
    fish_data.mean_body_width = mean_width

    return max_width, mean_width


def extract_decoupled_back_features(fish_data: "TopFishData") -> Tuple[float, float]:
    """
    直接基于 3D 骨架线 spine_3d 进行：
    1. 2D 弧长累加 (展平左右弯曲 Yaw)
    2. 1D 深度基线扣除去趋势 (剥离前后倾斜 Pitch)
    3. 1D 多项式拟合求解纯粹背部曲率 a 与净最大厚度 Delta_Z

    :param fish_data: 已填充 spine_2d 和 spine_3d 的 TopFishData 对象
    :return: (back_curvature, back_height)
    """
    spine_2d = fish_data.spine_2d
    spine_3d = fish_data.spine_3d

    if spine_2d is None or spine_3d is None:
        return 0.0, 0.0

    spine_2d_arr = np.array(spine_2d, dtype=np.float64)
    if len(spine_2d_arr) < 3 or len(spine_3d) < 3:
        return 0.0, 0.0

    # 直接从已有的 3D 骨架线提取物理深度 Z (三维坐标中的第 3 列)
    z_array = spine_3d[:, 2]


    # 步骤 1：2D 平面弧长累加 (展平 C/S 型左右摆尾弯曲 Yaw)
    diffs_2d = np.diff(spine_2d_arr, axis=0)
    step_distances = np.sqrt(np.sum(diffs_2d ** 2, axis=1))

    # s 为一维展平累加弧长向量，把弯曲的鱼拉直
    s = np.insert(np.cumsum(step_distances), 0, 0.0)

    # 如果已有 spine_3d，计算 3D 真实的累加弧长作为鱼体的实际物理总体长
    diffs_3d = np.diff(spine_3d, axis=0)
    step_distances_3d = np.sqrt(np.sum(diffs_3d ** 2, axis=1))
    fish_data.spine_length = float(np.sum(step_distances_3d))  # 赋值真实 3D 骨架弧长体长


    # 步骤 2：扣除倾斜趋势基线 (解耦斜上/斜下游动俯仰 Pitch)
    s_head, s_tail = s[0], s[-1]
    z_head, z_tail = z_array[0], z_array[-1]

    # 建立头尾连线的 1D 线性基线方程 Z_base(s)
    if abs(s_tail - s_head) > 1e-6:
        slope = (z_tail - z_head) / (s_tail - s_head)
        z_base = z_head + slope * (s - s_head)
    else:
        z_base = np.full_like(z_array, z_head)

    # 计算扣除基线后的相对背部隆起高度 Z_rel
    z_rel = z_array - z_base

    # -------------------------------------------------------------
    # 步骤 3：1D 抛物线拟合 Z_rel = a * s^2 + b * s + c 与平滑特征提取
    if len(s) >= 3:
        poly_coeffs = np.polyfit(s, z_rel, deg=2)
        a = poly_coeffs[0]

        # 特征 1: 背部曲率系数 a 的绝对值
        back_curvature = float(np.abs(a))

        # 特征 2: 利用拟合后的平滑曲线极值差求解净背部最大厚度 (天然滤除单点深度噪声)
        z_fit = np.polyval(poly_coeffs, s)
        back_height = float(np.max(z_fit) - np.min(z_fit))
    else:
        back_curvature = 0.0
        back_height = 0.0

    # 写回对象属性
    fish_data.back_curvature = back_curvature
    fish_data.back_height = back_height

    return back_curvature, back_height


def process_top_fish_list(fish_list: List["TopFishData"]) -> None:
    """
    批量处理顶视鱼数据列表的主接口函数。
    直接利用对象的 spine_3d 数据，逐一更新每条鱼的几何与解耦特征。

    :param fish_list: 包含 TopFishData 对象的列表
    :return: 计算并赋值好解耦特征后的 fish_list 列表
    """
    if not fish_list:
        return []

    for fish_data in fish_list:
        # 1. 利用 spine_3d 自动换算物理尺寸并提取法线切片体宽
        extract_body_width_from_3d_spine(fish_data)

        # 2. 2D 展平 + 1D 基线去趋势解耦背部曲率与净最大厚度
        extract_decoupled_back_features(fish_data)