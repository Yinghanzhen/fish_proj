from fish_weight_estimation.fish_camera_class import TopFishData,SideFishData
import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.morphology import skeletonize

class TopFishSkeletonExtractor:
    def __init__(self, blur_kernel_size: int = 5, min_component_size: int = 15):
        """
        :param blur_kernel_size: 高斯平滑核大小，用于抹平 YOLO 掩码上采样产生的阶梯锯齿
        :param min_component_size: 过滤小噪点骨架的最小像素数量
        """
        self.blur_ksize = blur_kernel_size
        self.min_comp_size = min_component_size

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """前处理：抹平上采样锯齿"""
        mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.copy()
        blurred = cv2.GaussianBlur(mask_u8, (self.blur_ksize, self.blur_ksize), 0)
        _, smoothed_mask = cv2.threshold(blurred, 127, 1, cv2.THRESH_BINARY)
        return smoothed_mask

    @staticmethod
    def _sort_skeleton_points(points: np.ndarray) -> np.ndarray:
        """沿骨架线几何邻接顺序排序无序像素点"""
        if len(points) <= 2:
            return points

        sorted_pts = [points[0]]
        remaining = list(points[1:])

        while remaining:
            last_pt = sorted_pts[-1]
            dists = np.linalg.norm(remaining - last_pt, axis=1)
            nearest_idx = np.argmin(dists)

            # 若最近点间距过大，说明到了当前段末尾
            if dists[nearest_idx] > 3.0:
                sorted_pts.extend(remaining)
                break

            sorted_pts.append(remaining.pop(nearest_idx))

        return np.array(sorted_pts)

    def extract_skeleton_with_padding(self, mask: np.ndarray) -> List[Optional[List[float]]]:
        """
        核心提取逻辑：从实例掩码中提取骨架点。
        若被遮挡断裂，自动按两轴投影最大值 max(Δx, Δy) 计算缺口长度并插入 [None, None] 占位符。
        """
        # 1. 前处理抗锯齿与骨架细化
        smooth_mask = self._preprocess_mask(mask)
        skeleton_bool = skeletonize(smooth_mask > 0)
        skeleton_u8 = skeleton_bool.astype(np.uint8)

        # 2. 连通域标记（检测是否被切断成多段）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton_u8, connectivity=8)

        parts = []
        for label in range(1, num_labels):  # label 0 为背景
            if stats[label, cv2.CC_STAT_AREA] < self.min_comp_size:
                continue  # 过滤孤立小噪点

            y_coords, x_coords = np.where(labels == label)
            points_xy = np.column_stack((x_coords, y_coords))
            parts.append(self._sort_skeleton_points(points_xy))

        # 情况 1: 无遮挡（只有 1 个完整的骨架连通域）
        if len(parts) == 1:
            return parts[0].astype(float).tolist()

        # 情况 2: 被遮挡断裂（存在 2 段及以上骨架）
        elif len(parts) >= 2:
            # 取主要的前后两段骨架
            part1, part2 = parts[0], parts[1]

            # 校准两段骨架的首尾方向，确保 part1 末端面向 part2 首端
            d_end_to_start = np.linalg.norm(part1[-1] - part2[0])
            d_end_to_end = np.linalg.norm(part1[-1] - part2[-1])
            d_start_to_start = np.linalg.norm(part1[0] - part2[0])
            d_start_to_end = np.linalg.norm(part1[0] - part2[-1])

            min_dist = min(d_end_to_start, d_end_to_end, d_start_to_start, d_start_to_end)

            if min_dist == d_end_to_end:
                part2 = part2[::-1]
            elif min_dist == d_start_to_start:
                part1 = part1[::-1]
            elif min_dist == d_start_to_end:
                part1 = part1[::-1]
                part2 = part2[::-1]

            # 取断口处的两个端点
            p1_end = part1[-1]  # 前半段断口 [x1, y1]
            p2_start = part2[0]  # 后半段断口 [x2, y2]

            # 核心投影计算：取 X 轴和 Y 轴缺失量的极值作为补全点数
            dx_missing = abs(p1_end[0] - p2_start[0])
            dy_missing = abs(p1_end[1] - p2_start[1])
            n_missing = int(np.round(max(dx_missing, dy_missing)))

            # 组装完整的列表：Part 1 + [None, None] * n_missing + Part 2
            spine_2d_padded: List[Optional[List[float]]] = []

            # 填充 Part 1 真实点
            for pt in part1:
                spine_2d_padded.append([float(pt[0]), float(pt[1])])

            # 自动填充计算出的 [None, None] 空位
            for _ in range(n_missing):
                spine_2d_padded.append([None, None])

            # 填充 Part 2 真实点
            for pt in part2:
                spine_2d_padded.append([float(pt[0]), float(pt[1])])

            return spine_2d_padded

        return []


    def process_and_update_fish(self, fish: TopFishData, mask: np.ndarray):
        """
        运行提取逻辑，并将提取结果直接写入输入的 TopFishData 实例中
        """
        result_spine = self.extract_skeleton_with_padding(mask)
        fish.spine_2d = result_spine if len(result_spine) > 0 else None