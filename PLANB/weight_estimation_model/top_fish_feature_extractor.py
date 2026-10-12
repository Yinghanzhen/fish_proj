from typing import List, Optional, Tuple
import cv2
import numpy as np
from PLANB.fish_camera_class import TopFishData

def extract_body_width_from_3d_spine(
    fish_data: TopFishData,
) -> Tuple[float, float]:
  """利用 2D 像素 Mask + 2D 像素骨架切片，并结合 3D 物理骨架距离自动计算物理体宽。
  :param fish_data: 已填充 spine_2d (像素), spine_3d (真实物理), mask 和 obscured 的 TopFishData 实例
  :return: (max_body_width, mean_body_width) 真实物理体宽
  """
  # 若鱼体被遮挡 (obscured 为 True) 或缺少基础数据，跳过体宽计算直接归零
  if (
      getattr(fish_data, "obscured", False)
      or fish_data.mask is None
      or fish_data.spine_2d is None
      or fish_data.spine_3d is None
  ):
    fish_data.max_body_width = 0.0
    fish_data.mean_body_width = 0.0
    return 0.0, 0.0

  spine_2d = np.array(fish_data.spine_2d, dtype=np.float32)  # 2D 像素坐标 (u, v)
  spine_3d = np.array(
      fish_data.spine_3d, dtype=np.float64
  )  # 3D 物理坐标 (X, Y, Z)

  # 1. 滤除包含 NaN 的无效骨架节点
  valid_mask = ~np.isnan(spine_2d).any(axis=1) & ~np.isnan(spine_3d).any(axis=1)
  valid_pts_2d = spine_2d[valid_mask]
  valid_pts_3d = spine_3d[valid_mask]

  num_pts = len(valid_pts_2d)
  if num_pts < 3:
    fish_data.max_body_width = 0.0
    fish_data.mean_body_width = 0.0
    return 0.0, 0.0

  # 将 mask 转换为 uint8 类型的二值化图 (0 和 255)
  mask_uint8 = fish_data.mask.astype(np.uint8) * 255
  widths_physical = []

  # 2. 计算相邻节点间的 2D 像素距离与 3D 真实物理距离
  diffs_2d = np.linalg.norm(np.diff(valid_pts_2d, axis=0), axis=1)  # (pixels)
  diffs_3d = np.linalg.norm(
      np.diff(valid_pts_3d, axis=0), axis=1
  )  # (物理单位 cm/mm)

  # 求解各段骨架微元的局域转换系数 (物理单位/pixel)
  valid_ratios = np.where(diffs_2d > 1e-3, diffs_3d / diffs_2d, 0.0)

  # 3. 遍历骨架节点 (除去首尾) 求切线法线与像素交点
  for i in range(1, num_pts - 1):
    pt_2d = valid_pts_2d[i]
    prev_pt_2d = valid_pts_2d[i - 1]
    next_pt_2d = valid_pts_2d[i + 1]

    # 求 2D 像素平面上的切线与法线向量
    tangent = next_pt_2d - prev_pt_2d
    norm = np.linalg.norm(tangent)
    if norm < 1e-5:
      continue

    # 2D 单位法线向量 (-dy, dx)
    normal_2d = np.array([-tangent[1], tangent[0]]) / norm

    # 沿法线方向在 Mask 上画线搜寻交点
    line_len = max(mask_uint8.shape) // 2
    p1 = (pt_2d + normal_2d * line_len).astype(np.int32)
    p2 = (pt_2d - normal_2d * line_len).astype(np.int32)

    line_mask = np.zeros_like(mask_uint8)
    cv2.line(line_mask, (p1[0], p1[1]), (p2[0], p2[1]), color=255, thickness=1)
    intersection = cv2.bitwise_and(mask_uint8, line_mask)

    # 统计交线覆盖的像素宽度 ΔW_pixel
    width_pixel = np.sum(intersection > 0)

    if width_pixel > 0:
      # 取当前节点前后的局域转换系数 (物理单位/pixel)
      local_scale = (
          valid_ratios[i - 1]
          if valid_ratios[i - 1] > 0
          else np.mean(valid_ratios[valid_ratios > 0])
          if np.any(valid_ratios > 0)
          else 1.0
      )

      # 像素体宽 × 局域物理比例 = 真实物理体宽
      widths_physical.append(width_pixel * local_scale)

  if not widths_physical:
    fish_data.max_body_width = 0.0
    fish_data.mean_body_width = 0.0
    return 0.0, 0.0

  max_width = float(np.max(widths_physical))
  mean_width = float(np.mean(widths_physical))

  # 直接写回 TopFishData 实例属性
  fish_data.max_body_width = max_width
  fish_data.mean_body_width = mean_width

  return max_width, mean_width


def extract_decoupled_back_features(
    fish_data: TopFishData,
) -> Tuple[float, float, float]:
  """提取 3D 骨架真实体长，并通过 2D 像素弧长展平 (解耦 Yaw) + 1D 线性去趋势 (解耦 Pitch)

  计算背部曲率与净最大物理厚度。

  :param fish_data: 已填充 spine_2d 和 spine_3d 的 TopFishData 实例
  :return: (spine_length, back_curvature, back_height)
  """
  spine_2d = fish_data.spine_2d
  spine_3d = fish_data.spine_3d

  if spine_2d is None or spine_3d is None:
    fish_data.spine_length = 0.0
    fish_data.back_curvature = 0.0
    fish_data.back_height = 0.0
    return 0.0, 0.0, 0.0

  spine_2d_arr = np.array(spine_2d, dtype=np.float64)  # 2D 像素坐标
  spine_3d_arr = np.array(spine_3d, dtype=np.float64)  # 3D 真实物理坐标

  # 过滤 NaN 节点（含之前的遮挡断裂补 None 转换）
  valid_mask = ~np.isnan(spine_2d_arr).any(axis=1) & ~np.isnan(
      spine_3d_arr
  ).any(axis=1)
  valid_2d = spine_2d_arr[valid_mask]
  valid_3d = spine_3d_arr[valid_mask]

  if len(valid_2d) < 3 or len(valid_3d) < 3:
    fish_data.spine_length = 0.0
    fish_data.back_curvature = 0.0
    fish_data.back_height = 0.0
    return 0.0, 0.0, 0.0


  # 步骤 1：自动计算并回写 3D 骨架线真实物理体长 (spine_length)
  diffs_3d = np.diff(valid_3d, axis=0)
  step_distances_3d = np.sqrt(np.sum(diffs_3d**2, axis=1))
  spine_length = float(np.sum(step_distances_3d))
  fish_data.spine_length = spine_length  # 回写实例

  # -------------------------------------------------------------
  # 步骤 2：2D 像素平面弧长累加 (展平左右弯曲 Yaw 摆尾)
  # -------------------------------------------------------------
  diffs_2d = np.diff(valid_2d, axis=0)
  step_distances_px = np.sqrt(np.sum(diffs_2d**2, axis=1))
  s = np.insert(np.cumsum(step_distances_px), 0, 0.0)  # 一维弧长坐标 s

  # 提取三维真实物理深度 Z 向量
  z_array = valid_3d[:, 2]

  # -------------------------------------------------------------
  # 步骤 3：扣除头尾连线基线 (解耦斜上/斜下游动 Pitch 俯仰倾斜)
  # -------------------------------------------------------------
  s_head, s_tail = s[0], s[-1]
  z_head, z_tail = z_array[0], z_array[-1]

  if abs(s_tail - s_head) > 1e-6:
    slope = (z_tail - z_head) / (s_tail - s_head)
    z_base = z_head + slope * (s - s_head)
  else:
    z_base = np.full_like(z_array, z_head)

  # 扣除基线后的相对背部隆起高度 Z_rel (物理单位 cm/mm)
  z_rel = z_array - z_base

  # -------------------------------------------------------------
  # 步骤 4：1D 抛物线拟合 Z_rel = a * s^2 + b * s + c 与特征计算
  # -------------------------------------------------------------
  poly_coeffs = np.polyfit(s, z_rel, deg=2)
  a = poly_coeffs[0]

  # 特征 1: 背部曲率系数 a 的绝对值
  back_curvature = float(np.abs(a))

  # 特征 2: 多项式拟合后的平滑极值差求解净背部最大物理厚度
  z_fit = np.polyval(poly_coeffs, s)
  back_height = float(np.max(z_fit) - np.min(z_fit))

  # 回写 TopFishData 实例属性
  fish_data.back_curvature = back_curvature
  fish_data.back_height = back_height

  return spine_length, back_curvature, back_height


def process_top_fish_list(fish_list: List[TopFishData]) -> None:
  """主入口函数：批量处理顶视鱼列表，原地更新每条鱼的所有几何特征。

  :param fish_list: 包含 TopFishData 对象的列表
  """
  if not fish_list:
    return

  for fish_data in fish_list:
    # 1. 提取物理体宽（只有当 fish_data.obscured 为 False 时才会有效提取，被遮挡则输出 0.0）
    extract_body_width_from_3d_spine(fish_data)

    # 2. 提取 3D 物理体长、解耦背部曲率与净背部最大物理厚度（即使遮挡，依然可以通过过滤有效 3D 骨架算出物理体长）
    extract_decoupled_back_features(fish_data)