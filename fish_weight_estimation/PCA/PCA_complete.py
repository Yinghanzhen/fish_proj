import pickle
from typing import Any, Dict, List, Union
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from fish_weight_estimation.fish_camera_class import TopFishData

# 模型全局缓存
_PCA_CACHE: Dict[str, Any] = {}

def resample_skeleton(pts: Union[np.ndarray, List[List[float]]], target_n: int = 100) -> np.ndarray:
    """按弧长均匀重采样至 target_n 个点"""
    pts = np.asarray(pts, dtype=np.float64)
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.insert(np.cumsum(dists), 0, 0)
    if cum[-1] == 0:
        return np.tile(pts[0], (target_n, 1))
    norm_cum = cum / cum[-1]
    return np.column_stack([
        interp1d(norm_cum, pts[:, d])(np.linspace(0, 1, target_n))
        for d in range(3)
    ])


def batch_repair_fish_skeletons(
    top_fishes: List[TopFishData],
    pca_model_path: str = "fish_pca_model.pkl",
    depth_threshold: float = 0.15,
) -> None:
    """批量对鱼实例列表进行 3D 骨架遮挡检测与 PCA 修复（假设头尾关键点恒不遮挡）"""
    # 1. 读取 PCA 模型（带缓存）
    if pca_model_path not in _PCA_CACHE:
        with open(pca_model_path, "rb") as f:
            _PCA_CACHE[pca_model_path] = pickle.load(f)
    target_n, pca = (
        _PCA_CACHE[pca_model_path]["target_n"],
        _PCA_CACHE[pca_model_path]["pca"],
    )

    # 2. 遍历修复
    for fish in top_fishes:
        if fish.spine_3d is None or len(fish.spine_3d) < 2:
            fish.is_occluded = False
            continue

        skel = resample_skeleton(fish.spine_3d, target_n=target_n)

        # 1. 深度突变检测（中间区域）
        diffs = np.diff(skel[:, 2])
        valid_mask = np.ones(target_n, dtype=bool)

        jumps_up = np.where(diffs > depth_threshold)[0]
        jumps_down = np.where(diffs < -depth_threshold)[0]

        for u in jumps_up:
            d_candidates = jumps_down[jumps_down > u]
            end_idx = d_candidates[0] + 1 if len(d_candidates) > 0 else target_n - 1
            valid_mask[u + 1 : end_idx] = False

        # 强制确保头尾节点有效
        valid_mask[0] = True
        valid_mask[-1] = True

        # 如果没有检测到遮挡，直接跳过
        if np.all(valid_mask):
            fish.is_occluded = False
            continue

        # 2. 姿态对齐（直接以 skel[0] 为头，skel[-1] 为尾）
        head_pt = skel[0].copy()
        skel_norm = skel - head_pt

        tail_vec = skel_norm[-1]
        norm_val = np.linalg.norm(tail_vec)

        if norm_val < 1e-6:
            fish.is_occluded = False
            continue

        unit_tail = tail_vec / norm_val
        target_vec = np.array([1.0, 0.0, 0.0])

        # 平行向量安全处理
        if np.abs(np.dot(unit_tail, target_vec)) > 0.999:
            rot = R.identity(1)
        else:
            rot, *_ = R.align_vectors([target_vec], [unit_tail])

        skel_aligned = rot.apply(skel_norm)

        # 3. PCA 最小二乘求解与缺失重建
        mask_flat = np.repeat(valid_mask, 3)

        alpha, _, _, _ = np.linalg.lstsq(
            pca.components_.T[mask_flat, :],
            skel_aligned.reshape(-1)[mask_flat] - pca.mean_[mask_flat],
            rcond=None,
        )

        recon = (pca.mean_ + pca.components_.T @ alpha).reshape(target_n, 3)

        # 仅替换被遮挡的中间节点
        skel_aligned[~valid_mask] = recon[~valid_mask]

        # 4. 逆变换还原写回
        fish.spine_3d = rot.inv().apply(skel_aligned) + head_pt
        fish.is_occluded = True