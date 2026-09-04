import os
import json
import glob

# 1. 类别字典定义 (请严格按索引对应 0 ~ 4)
CLASS_MAPPING = {
    "California_bass": 0,
    "grass_crap": 1,
    "Leopard_coral_ grouper": 2,  # 注意包含空格/拼写差异
    "Leopard_coral_grouper": 2,  # 容错处理
    "Pearl_grouper": 3,
    "tiger_grouper": 4
}

# 2. 关键点列表（顺序至关重要，决定模型学习的通道顺序）
# [up, bottom, tail, head]
KEYPOINT_ORDER = ["up", "bottom", "tail", "head"]


def convert_anylabeling_json_to_yolo(json_file, output_txt_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data.get("imageWidth", 1280)
    img_h = data.get("imageHeight", 720)

    # 按 group_id 分组整理 targets
    # 数据结构: { group_id: {"bbox_label": str, "points": [[x1,y1,y2,x2]], "kpts": {label: [x,y]}} }
    groups = {}

    for shape in data.get("shapes", []):
        group_id = shape.get("group_id", 0)
        label = shape.get("label", "").strip()
        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])

        if group_id not in groups:
            groups[group_id] = {"bbox_label": None, "box_points": None, "kpts": {}}

        # 错别字容错：处理标注时可能写成 "buttom" 的情况
        if label.lower() == "buttom":
            label = "bottom"

        # 提取边界框 (rectangle / polygon)
        if shape_type in ["rectangle", "polygon"] and label in CLASS_MAPPING:
            groups[group_id]["bbox_label"] = label
            groups[group_id]["box_points"] = points

        # 提取关键点 (point)
        elif shape_type == "point" and label in KEYPOINT_ORDER:
            groups[group_id]["kpts"][label] = points[0]

    yolo_lines = []

    for gid, target in groups.items():
        bbox_label = target["bbox_label"]
        box_pts = target["box_points"]

        # 如果没有找到框，则跳过该组
        if bbox_label is None or box_pts is None:
            continue

        class_id = CLASS_MAPPING[bbox_label]

        # 计算 Bbox 的 [xmin, ymin, xmax, ymax]
        xs = [p[0] for p in box_pts]
        ys = [p[1] for p in box_pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # 归一化为 YOLO 的 cx, cy, w, h
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # 构建关键点文本 [px, py, v]
        kpt_str_list = []
        for kpt_name in KEYPOINT_ORDER:
            if kpt_name in target["kpts"]:
                kx, ky = target["kpts"][kpt_name]
                # 坐标归一化，v=2 代表可见标注
                nkx = kx / img_w
                nky = ky / img_h
                kpt_str_list.append(f"{nkx:.6f} {nky:.6f} 2")
            else:
                # 若缺失该关键点，充填 0 0 0 (v=0 代表未标注)
                kpt_str_list.append("0.000000 0.000000 0")

        kpts_str = " ".join(kpt_str_list)
        line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpts_str}"
        yolo_lines.append(line)

    # 写入文件
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines))


def batch_convert(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_list = glob.glob(os.path.join(json_dir, "*.json"))

    print(f"找到 {len(json_list)} 个 JSON 文件，开始转换...")
    for json_path in json_list:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        convert_anylabeling_json_to_yolo(json_path, txt_path)

    print("转换完成！标注已成功生成至:", output_dir)


if __name__ == "__main__":
    # 配置你的 JSON 输入路径与 TXT 保存路径
    JSON_DIR = r"C:\Users\ying'han'zhen\Desktop\data\json"  # AnyLabeling 导出的 JSON 文件夹
    OUTPUT_DIR = r"C:\Users\ying'han'zhen\Desktop\data\labels"  # 输出的 YOLO txt 标注文件夹

    batch_convert(JSON_DIR, OUTPUT_DIR)