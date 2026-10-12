from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from collections import deque


# ==================== 推理 ====================
model = YOLO(r"D:\Project\fish_planB\runs\segment\train2\weights\best.pt")
IMG_PATH = r"D:\Project\fish_planB\data\yolo_seg\images\val\IMG20260923144814.jpg"
results = model(IMG_PATH, conf=0.1, iou=0.1)
r = results[0]

annotated = r.plot()
cv2.imwrite("result.jpg", annotated)
print("整体可视化已保存: result.jpg")

img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

n = len(r.boxes) if r.boxes is not None else 0
print(f"检测到 {n} 条鱼")
if n == 0:
    print("没有检测到鱼，退出")
    exit()

boxes = r.boxes.xyxy.cpu().numpy()
confs = r.boxes.conf.cpu().numpy()
masks = r.masks.data.cpu().numpy()


# ==================== 骨架线提取 ====================
def extract_skeleton(mask_binary):
    skel = skeletonize(mask_binary.astype(bool))
    return skel.astype(np.uint8)


def extract_main_path(skeleton):
    ys, xs = np.nonzero(skeleton)
    if len(xs) == 0:
        return []

    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    neighbor_count = convolve(skeleton.astype(int), kernel, mode="constant")
    endpoints_mask = (skeleton > 0) & (neighbor_count == 11)
    endpoints = np.stack(np.nonzero(endpoints_mask), axis=1)

    if len(endpoints) < 2:
        return [[int(x), int(y)] for y, x in zip(ys, xs)]

    skeleton_set = set(zip(ys.tolist(), xs.tolist()))

    def bfs(start):
        visited = {start: None}
        queue = deque([start])
        farthest = start
        while queue:
            cur = queue.popleft()
            farthest = cur
            cy, cx = cur
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nxt = (cy + dy, cx + dx)
                    if nxt in skeleton_set and nxt not in visited:
                        visited[nxt] = cur
                        queue.append(nxt)
        return farthest, visited

    start = tuple(endpoints[0])
    end_a, _ = bfs(start)
    end_b, visited = bfs(end_a)

    path = []
    cur = end_b
    while cur is not None:
        path.append(cur)
        cur = visited[cur]
    path.reverse()
    return [[int(x), int(y)] for y, x in path]


# ==================== 用圆内 mask 数量判断头尾 ====================
def circle_mask_count(mask, cx, cy, radius=50):
    """
    以 (cx, cy) 为圆心，半径 radius，统计圆内 mask 的像素数。
    """
    H, W = mask.shape
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    circle = dist2 <= radius ** 2
    return int(np.logical_and(circle, mask).sum())


def decide_head_tail(mask, path, radius=120):
    """
    用骨架两端点的「圆内 mask 数量」判断头尾。
    多的 = 头，少的 = 尾。
    返回：(head_xy, tail_xy, cnt_head, cnt_tail)
    """
    if len(path) < 2:
        return None, None, 0, 0

    p_first = path[0]     # [x, y]
    p_last = path[-1]     # [x, y]

    cnt_first = circle_mask_count(mask, p_first[0], p_first[1], radius)
    cnt_last = circle_mask_count(mask, p_last[0], p_last[1], radius)

    if cnt_first >= cnt_last:
        head = p_first
        tail = p_last
        cnt_head, cnt_tail = cnt_first, cnt_last
    else:
        head = p_last
        tail = p_first
        cnt_head, cnt_tail = cnt_last, cnt_first

    return head, tail, cnt_head, cnt_tail


# ==================== 逐条鱼可视化 ====================
RADIUS = 120

for i in range(n):
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    mask = masks[i]
    mask_full = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5

    overlay = img.copy()
    overlay[mask_full] = (255, 0, 0)
    vis1 = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    x1, y1, x2, y2 = boxes[i].astype(int)
    cv2.rectangle(vis1, (x1, y1), (x2, y2), (0, 255, 0), 3)

    axes[0].imshow(vis1)
    axes[0].set_title(f"Fish {i} | conf={confs[i]:.3f}")
    axes[0].axis("off")

    # ---- 子图 2：mask ----
    axes[1].imshow(mask_full, cmap="gray")
    axes[1].set_title(f"Mask {i} | pixels={int(mask_full.sum())}")
    axes[1].axis("off")

    # ---- 子图 3：骨架 ----
    skeleton = extract_skeleton(mask_full)
    axes[2].imshow(skeleton, cmap="gray")
    axes[2].set_title(f"Skeleton {i} | pixels={int(skeleton.sum())}")
    axes[2].axis("off")

    # ---- 子图 4：叠加 + 头尾判断 ----
    vis4 = img.copy()
    overlay = img.copy()
    overlay[mask_full] = (255, 0, 0)
    vis4 = cv2.addWeighted(vis4, 0.6, overlay, 0.4, 0)

    # 骨架线（绿色）
    skel_ys, skel_xs = np.nonzero(skeleton)
    vis4[skel_ys, skel_xs] = (0, 255, 0)

    # 主路径（蓝色加粗）
    path = extract_main_path(skeleton)
    for (px, py) in path:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < W and 0 <= ny < H:
                    vis4[ny, nx] = (0, 0, 255)

    # 头尾判断
    head_xy, tail_xy, cnt_head, cnt_tail = decide_head_tail(
        mask_full, path, radius=RADIUS
    )

    # 画两个圆 + 标记
    if head_xy is not None:
        # 头：绿色圆 + "HEAD" 文字
        cv2.circle(vis4, head_xy, RADIUS, (0, 255, 0), 3)
        cv2.putText(vis4, f"HEAD ({cnt_head})",
                    (head_xy[0] + 10, head_xy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 尾：黄色圆 + "TAIL" 文字
        cv2.circle(vis4, tail_xy, RADIUS, (255, 255, 0), 3)
        cv2.putText(vis4, f"TAIL ({cnt_tail})",
                    (tail_xy[0] + 10, tail_xy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # 画端点
        cv2.circle(vis4, head_xy, 5, (0, 255, 0), -1)
        cv2.circle(vis4, tail_xy, 5, (255, 255, 0), -1)

    axes[3].imshow(vis4)
    axes[3].set_title(
        f"Overlay {i} | HEAD={cnt_head} px, TAIL={cnt_tail} px"
    )
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(f"fish_{i}_head_tail.png", dpi=100)
    plt.close()

    # 打印
    print(f"\nFish {i}: mask={int(mask_full.sum())} px, "
          f"skeleton={int(skeleton.sum())} px, "
          f"main_path={len(path)} pts, "
          f"conf={confs[i]:.3f}")
    if head_xy is not None:
        print(f"       HEAD = {head_xy}, 圆内 mask = {cnt_head} px")
        print(f"       TAIL = {tail_xy}, 圆内 mask = {cnt_tail} px")

    # 骨架弧长
    if len(path) >= 2:
        pts = np.array(path, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        arc_len = float(seg_len.sum())
        print(f"       骨架弧长 = {arc_len:.2f} 像素")