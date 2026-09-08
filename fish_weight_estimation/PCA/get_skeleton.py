import numpy as np
from skimage.morphology import skeletonize
from typing import List, Optional, Tuple

def extract_skeleton(
        mask: np.ndarray,
        head_kpt: Tuple[float, float],
        tail_kpt: Tuple[float, float],
        pixel_step_threshold: float = 1.5  # 相邻像素欧氏距离阈值，超过则判定为断裂并填充对应数量的 None
) -> List[Optional[List[float]]]:
    """
    提取原始 2D 骨架线序列。
    按从头到尾的物理顺序连接像素，并在断裂处按真实像素距离精准插入对应数量的 None 实例。
    返回:
        List[Optional[List[float]]]: 包含所有原始骨架点 [x, y] 及断裂处 None 值的完整列表。
    """
    # 1. 细化提取骨架像素
    skel = skeletonize(mask)
    pts = np.column_stack(np.where(skel))[:, [1, 0]].astype(np.float64)  # (y,x) -> (x,y)

    if len(pts) == 0:
        return []

    # 2. 查找与真实头尾 keypoints 最近的骨架端点
    head_arr = np.array(head_kpt, dtype=np.float64)
    tail_arr = np.array(tail_kpt, dtype=np.float64)

    head_idx = np.argmin(np.linalg.norm(pts - head_arr, axis=1))
    tail_idx = np.argmin(np.linalg.norm(pts - tail_arr, axis=1))

    # 3. 贪心路径追踪：从头到尾对骨架像素进行物理顺序排序
    ordered_pts = [pts[head_idx]]
    visited = {head_idx}
    curr = head_idx

    while curr != tail_idx:
        dists = np.linalg.norm(pts - pts[curr], axis=1)
        dists[list(visited)] = np.inf
        nxt = np.argmin(dists)

        if np.isinf(dists[nxt]):
            break  # 无法继续访问下一个点，终止

        visited.add(nxt)
        ordered_pts.append(pts[nxt])
        curr = nxt

    # 补齐真实的头尾 Keypoint 坐标
    ordered_pts = [head_arr] + ordered_pts + [tail_arr]

    # 4. 依次把像素点和断裂处对应数量的 None 填入实例列表中
    raw_skeleton_with_none: List[Optional[List[float]]] = []

    for i in range(len(ordered_pts) - 1):
        p1 = ordered_pts[i]
        p2 = ordered_pts[i + 1]

        # 存入当前有效像素点坐标 [x, y]
        raw_skeleton_with_none.append(p1.tolist())

        # 计算相邻两点间的欧氏距离
        dist = np.linalg.norm(p2 - p1)

        # 若距离大于步长阈值（比如 1.5 像素），说明存在遮挡断裂
        if dist > pixel_step_threshold:
            # 算出缺失的像素个数，按数量压入 None
            num_missing = int(np.round(dist)) - 1
            for _ in range(num_missing):
                raw_skeleton_with_none.append(None)

    # 存入最后一个尾部点
    raw_skeleton_with_none.append(ordered_pts[-1].tolist())

    return raw_skeleton_with_none