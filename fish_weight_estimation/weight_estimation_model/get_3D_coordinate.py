from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple, Union
import numpy as np
from fish_weight_estimation.fish_camera_class import TopFishData,SideFishData,CameraParams

# 2D 像素到 3D 空间的底层解算 (含水下折射校正)
def pixel_depth_to_3d_refracted(
    u: float,
    v: float,
    depth_raw_mm: float,
    cam: CameraParams,
    is_side_view: bool = False,
) -> Tuple[float, float, float]:
    """将单个 2D 像素 (u, v) 结合原始深度 Z_raw 解算为折射校正后的 3D 坐标 (X, Y, Z)"""
    if depth_raw_mm <= 0 or math.isnan(depth_raw_mm):
        return (0.0, 0.0, 0.0)

    # 1. 空气中的入射角
    dx = (u - cam.cx) / cam.fx
    dy = (v - cam.cy) / cam.fy
    tan_theta_air = math.sqrt(dx**2 + dy**2)
    sin_theta_air = (
        tan_theta_air / math.sqrt(1 + tan_theta_air**2)
        if tan_theta_air > 0
        else 0.0
    )

    # 2. 折射角计算 (斯涅尔定律)
    sin_theta_water = min(
        1.0, max(-1.0, sin_theta_air * (cam.n_air / cam.n_water))
    )
    cos_theta_water = math.sqrt(1.0 - sin_theta_water**2)
    tan_theta_water = (
        sin_theta_water / cos_theta_water if cos_theta_water > 1e-6 else 0.0
    )

    if is_side_view and cam.glass_thick_mm > 0:
        sin_theta_glass = min(
            1.0, max(-1.0, sin_theta_air * (cam.n_air / cam.n_glass))
        )
        cos_theta_glass = math.sqrt(1.0 - sin_theta_glass**2)
        tan_theta_glass = (
            sin_theta_glass / cos_theta_glass if cos_theta_glass > 1e-6 else 0.0
        )
    else:
        tan_theta_glass = 0.0

    # 3. Z 轴物理实际深度与径向总偏移计算
    if not is_side_view:
        # 俯视 (空气 -> 水)
        z_sub_apparent = max(0.0, depth_raw_mm - cam.air_dist_mm)
        z_sub_true = z_sub_apparent * (cam.n_water / cam.n_air)
        Z_true = cam.air_dist_mm + z_sub_true
        r_total = (
            cam.air_dist_mm * tan_theta_air + z_sub_true * tan_theta_water
        )
    else:
        # 侧视 (空气 -> 玻璃 -> 水)
        z_sub_apparent = max(
            0.0, depth_raw_mm - cam.air_dist_mm - cam.glass_thick_mm
        )
        z_sub_true = z_sub_apparent * (cam.n_water / cam.n_air)
        Z_true = cam.air_dist_mm + cam.glass_thick_mm + z_sub_true
        r_total = (
            cam.air_dist_mm * tan_theta_air
            + cam.glass_thick_mm * tan_theta_glass
            + z_sub_true * tan_theta_water
        )

    # 4. 映射到 X, Y 轴
    phi = math.atan2(dy, dx)
    X_true = r_total * math.cos(phi)
    Y_true = r_total * math.sin(phi)

    return (float(X_true), float(Y_true), float(Z_true))


def get_depth_at_pixel(
    depth_map: np.ndarray, u: float, v: float, window_size: int = 3
) -> float:
    """提取像素 (u, v) 处的深度，小邻域中值滤波防空洞"""
    h, w = depth_map.shape[:2]
    cx, cy = int(round(u)), int(round(v))

    r = window_size // 2
    y_min, y_max = max(0, cy - r), min(h, cy + r + 1)
    x_min, x_max = max(0, cx - r), min(w, cx + r + 1)

    patch = depth_map[y_min:y_max, x_min:x_max]
    valid_mask = (patch > 0) & (~np.isnan(patch))

    if np.any(valid_mask):
        return float(np.median(patch[valid_mask]))
    return 0.0


# 4. 核心功能函数 (只负责填充 3D 坐标)
def process_fish_3d_data(
    top_fishes: Optional[List[TopFishData]] = None,
    side_fishes: Optional[List[SideFishData]] = None,
    top_depth_map: Optional[np.ndarray] = None,
    side_depth_map: Optional[np.ndarray] = None,
    top_cam_params: Optional[Union[CameraParams, dict]] = None,
    side_cam_params: Optional[Union[CameraParams, dict]] = None,
) -> Tuple[List[TopFishData], List[SideFishData]]:
    """读取深度图与相机参数，自动解算并填入 top_fishes 与 side_fishes 中的所有 3D 关键点坐标与 3D 骨架线。"""

    def parse_params(p):
        if isinstance(p, dict):
            return CameraParams(**p)
        return p

    top_cam = parse_params(top_cam_params)
    side_cam = parse_params(side_cam_params)

    # 1. 填充俯视鱼类的 3D 坐标 (head_kpt_3d, tail_kpt_3d, spine_3d)
    if top_fishes and top_depth_map is not None and top_cam is not None:
        for fish in top_fishes:
            # 鱼头 3D
            if fish.head_kpt_2d:
                u, v = fish.head_kpt_2d
                d_head = get_depth_at_pixel(top_depth_map, u, v)
                fish.head_kpt_3d = pixel_depth_to_3d_refracted(
                    u, v, d_head, top_cam, is_side_view=False
                )

            # 鱼尾 3D
            if fish.tail_kpt_2d:
                u, v = fish.tail_kpt_2d
                d_tail = get_depth_at_pixel(top_depth_map, u, v)
                fish.tail_kpt_3d = pixel_depth_to_3d_refracted(
                    u, v, d_tail, top_cam, is_side_view=False
                )

            # 骨架线点阵 3D (spine_3d)
            if fish.spine_2d:
                spine_3d_list = []
                for pt_2d in fish.spine_2d:
                    if pt_2d is not None and len(pt_2d) >= 2:
                        u, v = pt_2d[0], pt_2d[1]
                        d = get_depth_at_pixel(top_depth_map, u, v)
                        p_3d = pixel_depth_to_3d_refracted(
                            u, v, d, top_cam, is_side_view=False
                        )
                        spine_3d_list.append(p_3d)
                    else:
                        spine_3d_list.append([0.0, 0.0, 0.0])

                fish.spine_3d = np.array(spine_3d_list, dtype=np.float64)

    # 2. 填充侧视鱼类的 3D 坐标 (top_kpt_3d, bottom_kpt_3d)
    if side_fishes and side_depth_map is not None and side_cam is not None:
        for fish in side_fishes:
            # 背部点 3D
            if fish.top_kpt_2d:
                u, v = fish.top_kpt_2d
                d_top = get_depth_at_pixel(side_depth_map, u, v)
                fish.top_kpt_3d = pixel_depth_to_3d_refracted(
                    u, v, d_top, side_cam, is_side_view=True
                )

            # 腹部点 3D
            if fish.bottom_kpt_2d:
                u, v = fish.bottom_kpt_2d
                d_bottom = get_depth_at_pixel(side_depth_map, u, v)
                fish.bottom_kpt_3d = pixel_depth_to_3d_refracted(
                    u, v, d_bottom, side_cam, is_side_view=True
                )

    return top_fishes, side_fishes


# 5. 测试运行示例
if __name__ == "__main__":
    # 相机配置
    top_cam_cfg = CameraParams(
        fx=615.0, fy=615.0, cx=320.0, cy=240.0, air_dist_mm=400.0
    )
    side_cam_cfg = CameraParams(
        fx=620.0,
        fy=620.0,
        cx=320.0,
        cy=240.0,
        air_dist_mm=200.0,
        glass_thick_mm=10.0,
    )

    # 深度图与测试数据
    top_depth = np.full((480, 640), 650.0, dtype=np.float32)
    side_depth = np.full((480, 640), 550.0, dtype=np.float32)

    top_fish = TopFishData(
        fish_id=1,
        bbox=(100, 200, 500, 300),
        head_kpt_2d=(150, 250),
        tail_kpt_2d=(450, 250),
        spine_2d=[[150, 250], [250, 255], [350, 245], [450, 250]],
    )
    side_fish = SideFishData(
        fish_id=101,
        bbox=(200, 100, 400, 350),
        top_kpt_2d=(300, 120),
        bottom_kpt_2d=(300, 320),
    )

    # 执行解算
    proc_top, proc_side = process_fish_3d_data(
        top_fishes=[top_fish],
        side_fishes=[side_fish],
        top_depth_map=top_depth,
        side_depth_map=side_depth,
        top_cam_params=top_cam_cfg,
        side_cam_params=side_cam_cfg,
    )

    # 验证填充结果
    print("俯视 spine_3d 矩阵 Shape:", proc_top[0].spine_3d.shape)
    print("俯视 spine_3d 第一个点:", proc_top[0].spine_3d[0])
    print("侧视 top_kpt_3d 坐标:", proc_side[0].top_kpt_3d)