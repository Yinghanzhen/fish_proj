import cv2
import numpy as np
from typing import List, Optional, Tuple
from fish_weight_estimation.fish_class import TopFishData, SideFishData

class StereoSpine3DReconstructor:
    """
    双目 3D 骨架重建器：
    以左图为主视图，自动在右图进行极线匹配，解算 3D 空间坐标并回写至 TopFishData 实例。
    """
    def __init__(self, K1: np.ndarray, D1: np.ndarray,
                 K2: np.ndarray, D2: np.ndarray,
                 R: np.ndarray, T: np.ndarray,
                 patch_size: int = 7,
                 max_disparity: int = 150,
                 match_threshold: float = 0.6):
        """
        :param K1, D1: 左相机内参矩阵 (3x3) 与畸变系数
        :param K2, D2: 右相机内参矩阵 (3x3) 与畸变系数
        :param R, T: 双目相机外参 (左相机到右相机的旋转矩阵和平移向量)
        :param patch_size: 匹配小方块半径 (窗口大小为 (2*patch_size+1)^2)
        :param max_disparity: 右图水平沿极线搜索的最大视差范围 (像素)
        :param match_threshold: NCC 模板匹配置信度阈值 (0~1)
        """
        self.K1, self.D1 = K1, D1
        self.K2, self.D2 = K2, D2
        self.patch_size = patch_size
        self.max_disparity = max_disparity
        self.match_threshold = match_threshold

        # 构建左右相机的 3x4 投影矩阵 P1, P2 (以左相机坐标系为世界坐标系原点)
        self.P1 = np.hstack((self.K1, np.zeros((3, 1))))
        RT = np.hstack((R, T.reshape(3, 1)))
        self.P2 = self.K2 @ RT

    def _find_matching_point(self, img_left: np.ndarray, img_right: np.ndarray,
                             pt_left: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """底层方法：输入左图单点，利用极线约束自动在右图搜寻最佳匹配点"""
        xl, yl = int(pt_left[0]), int(pt_left[1])
        h, w = img_left.shape[:2]

        # 边界安全检查
        if (xl - self.patch_size < 0 or xl + self.patch_size >= w or
                yl - self.patch_size < 0 or yl + self.patch_size >= h):
            return None

        # 1. 截取左图局部模板
        template = img_left[yl - self.patch_size: yl + self.patch_size + 1,
                   xl - self.patch_size: xl + self.patch_size + 1]

        # 2. 沿右图极线（水平扫描线）截取搜索带
        xr_min = max(self.patch_size, xl - self.max_disparity)
        xr_max = xl
        if xr_max - xr_min < self.patch_size:
            return None

        search_strip = img_right[yl - self.patch_size: yl + self.patch_size + 1,
                       xr_min - self.patch_size: xr_max + self.patch_size + 1]

        if template.shape[0] > search_strip.shape[0] or template.shape[1] > search_strip.shape[1]:
            return None

        # 3. NCC (归一化互相关) 模板匹配
        res = cv2.matchTemplate(search_strip, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # 4. 超过阈值则认为找到对应点
        if max_val >= self.match_threshold:
            match_x = float(xr_min + max_loc[0])
            match_y = float(yl)  # 已极线校正前提下 y 轴平行；未严格校正可使用 yl + max_loc[1]
            return (match_x, match_y)

        return None

    def _triangulate_batch(self, pts_left: List[Tuple[float, float]],
                           pts_right: List[Tuple[float, float]]) -> np.ndarray:
        """底层方法：调用 OpenCV 三角测量算子批量计算 3D 物理坐标 (X, Y, Z)"""
        pts_l = np.array(pts_left, dtype=np.float64).reshape(-1, 1, 2)
        pts_r = np.array(pts_right, dtype=np.float64).reshape(-1, 1, 2)

        # 相机去畸变与坐标归一化
        undist_l = cv2.undistortPoints(pts_l, self.K1, self.D1, P=self.K1).reshape(-1, 2).T
        undist_r = cv2.undistortPoints(pts_r, self.K2, self.D2, P=self.K2).reshape(-1, 2).T

        # 双目三角测量
        pts_4d = cv2.triangulatePoints(self.P1, self.P2, undist_l, undist_r)

        # 齐次坐标转 3D 真实物理坐标 (X/W, Y/W, Z/W)
        pts_3d = (pts_4d[:3, :] / pts_4d[3, :]).T
        return pts_3d

    def process_fish_list(self, fish_list: List[TopFishData],
                          img_left: np.ndarray,
                          img_right: np.ndarray) -> List[TopFishData]:
        """
        主入口：传入鱼实例列表及左右图，自动提取右图坐标、计算 3D 坐标并写回原实例。

        :param fish_list: 包含左图 2D 数据的 TopFishData 实例列表
        :param img_left: 左相机图像 (灰度图或彩色图)
        :param img_right: 右相机图像
        :return: 处理完成后的 fish_list (写回了 head_kpt_3d, tail_kpt_3d, spine_3d, spine_length)
        """
        # 转为单通道灰度图提升模板匹配效率与鲁棒性
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if img_left.ndim == 3 else img_left
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) if img_right.ndim == 3 else img_right

        for fish in fish_list:
            # 1. 鱼头与鱼尾 3D 坐标解算
            for kpt_2d_attr, kpt_3d_attr in [('head_kpt_2d', 'head_kpt_3d'), ('tail_kpt_2d', 'tail_kpt_3d')]:
                pt_2d = getattr(fish, kpt_2d_attr)
                if pt_2d is not None:
                    match_r = self._find_matching_point(gray_left, gray_right, pt_2d)
                    if match_r is not None:
                        pt_3d = self._triangulate_batch([pt_2d], [match_r])[0]
                        setattr(fish, kpt_3d_attr, tuple(pt_3d))

            # 2. 骨架线 (Spine 2D) 自动匹配与 3D 解算
            if not fish.spine_2d:
                continue

            N_pts = len(fish.spine_2d)
            spine_3d_array = np.full((N_pts, 3), np.nan, dtype=np.float64)

            matched_pts_l = []
            matched_pts_r = []
            valid_indices = []

            # 遍历左图 2D 骨架点，跳过预留的 [None, None] 遮挡点
            for idx, pt in enumerate(fish.spine_2d):
                if pt is None or pt == [None, None] or np.isnan(pt[0]):
                    continue  # 保留为 np.nan

                match_pt_r = self._find_matching_point(gray_left, gray_right, (pt[0], pt[1]))
                if match_pt_r is not None:
                    matched_pts_l.append(pt)
                    matched_pts_r.append(match_pt_r)
                    valid_indices.append(idx)

            # 批量三角测量算出的有效 3D 坐标，精准写回对应的数组索引
            if len(matched_pts_l) > 0:
                calc_3d = self._triangulate_batch(matched_pts_l, matched_pts_r)
                for i, orig_idx in enumerate(valid_indices):
                    spine_3d_array[orig_idx] = calc_3d[i]

            # 写回实例属性
            fish.spine_3d = spine_3d_array

            # 3. 统计现有已知无遮挡 3D 骨架片段的累计体长
            valid_mask = ~np.isnan(spine_3d_array[:, 0])
            valid_3d_pts = spine_3d_array[valid_mask]
            if len(valid_3d_pts) >= 2:
                diffs = np.diff(valid_3d_pts, axis=0)
                segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
                fish.spine_length = float(np.sum(segment_lengths))

        return fish_list
