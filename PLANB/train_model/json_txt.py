import os
import json
import glob
import shutil
import numpy as np


# ================== 配置 ==================
SRC_DIR = r"D:\Project\fish_planB\data\top_fish_data"      # JSON + 图片目录
OUT_DIR = r"D:\Project\fish_planB\data\yolo_seg"           # 输出目录
VAL_RATIO = 0.2                                            # 验证集比例
# ==========================================


def convert_one(json_path, out_img_dir, out_lbl_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    orig_h = data["imageHeight"]
    orig_w = data["imageWidth"]
    img_name = os.path.splitext(os.path.basename(json_path))[0]

    # 找图片
    img_path = os.path.join(SRC_DIR, data.get("imagePath", ""))
    if not os.path.exists(img_path):
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = json_path.rsplit(".", 1)[0] + ext
            if os.path.exists(p):
                img_path = p
                break
    if not os.path.exists(img_path):
        print(f"[SKIP] 找不到图片: {json_path}")
        return False

    # 提取所有 polygon，写 YOLO-seg 行
    lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue

        arr = np.array(pts, dtype=np.float32)
        arr[:, 0] = np.clip(arr[:, 0] / orig_w, 0.0, 1.0)
        arr[:, 1] = np.clip(arr[:, 1] / orig_h, 0.0, 1.0)
        coords = " ".join(f"{v:.6f}" for v in arr.flatten())
        lines.append(f"0 {coords}")

    if len(lines) == 0:
        print(f"[SKIP] 无 polygon: {json_path}")
        return False

    shutil.copy(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))
    with open(os.path.join(out_lbl_dir, img_name + ".txt"), "w") as f:
        f.write("\n".join(lines))

    return True


def main():
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.json")))
    print(f"找到 {len(json_files)} 个 JSON")
    if len(json_files) == 0:
        return

    # 划分 train/val
    n_val = max(1, int(len(json_files) * VAL_RATIO))
    val_files = json_files[:n_val]
    train_files = json_files[n_val:]

    print(f"train={len(train_files)}, val={len(val_files)}")

    n_ok = 0
    for split, files in [("train", train_files), ("val", val_files)]:
        img_out = os.path.join(OUT_DIR, "images", split)
        lbl_out = os.path.join(OUT_DIR, "labels", split)
        for jp in files:
            if convert_one(jp, img_out, lbl_out):
                n_ok += 1

    print(f"转换完成: {n_ok} 张")

    # 生成 data.yaml
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""path: {OUT_DIR}
train: images/train
val: images/val

nc: 1
names:
  0: fish
""")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()