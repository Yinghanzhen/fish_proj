import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO


# ==================== 配置 ====================
MODEL_PATH = r"D:\Project\fish_planB\runs\segment\train\weights\best.pt"
INPUT_DIR  = r"D:\Project\fish_planB\data\top_fish_data"
OUTPUT_DIR = r"D:\Project\fish_planB\data\fish_masks"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.2


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    model = YOLO(MODEL_PATH)
    print(f"模型加载完成: {MODEL_PATH}")

    # 找所有图片
    img_files = []
    for ext in IMG_EXTS:
        img_files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        img_files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))
    img_files = sorted(set(img_files))

    print(f"找到 {len(img_files)} 张图片")
    if len(img_files) == 0:
        print("没有图片，退出")
        return

    total_fish = 0

    for idx, img_path in enumerate(img_files):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 读原图，用于获得 H, W
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] 读不到: {img_path}")
            continue
        H, W = img.shape[:2]

        # 推理
        results = model(
            img_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )
        r = results[0]

        n = len(r.boxes) if r.boxes is not None else 0
        if n == 0:
            print(f"[{idx+1}/{len(img_files)}] {img_name}: 未检测到鱼")
            continue

        boxes = r.boxes.xyxy.cpu().numpy()   # [N, 4]
        confs = r.boxes.conf.cpu().numpy()   # [N]
        masks = r.masks.data.cpu().numpy()   # [N, h, w] 下采样

        # 逐条鱼保存 mask
        for i in range(n):
            # mask 上采样到原图尺寸，二值化
            mask_full = cv2.resize(
                masks[i], (W, H), interpolation=cv2.INTER_NEAREST
            ) > 0.5

            # 转成 0/255 的黑白图
            mask_uint8 = (mask_full.astype(np.uint8)) * 255

            # 文件名：原图名_fish{i}.png
            out_name = f"{img_name}_fish{i}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, mask_uint8)

        total_fish += n
        print(f"[{idx+1}/{len(img_files)}] {img_name}: 检测到 {n} 条鱼，已保存")

    print(f"处理图片: {len(img_files)} 张")
    print(f"保存 mask: {total_fish} 张")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()