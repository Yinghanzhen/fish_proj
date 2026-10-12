import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class JSONPoseSegDataset(Dataset):
    """
    读 JSON，返回官方 loss 需要的格式：
      img        : [3, H, W] float
      cls        : [N, 1]  类别
      bboxes     : [N, 4]  xywh 归一化
      keypoints  : [N, K, 3] 归一化
      masks      : [N, H, W] 0/1 float
      batch_idx  : [N]  每张图里目标的 batch 索引，单图训练时全 0
    """

    def __init__(self, json_dir, img_size=640, kpt_names=("head", "tail")):
        super().__init__()
        self.img_size = img_size
        self.kpt_names = [k.lower() for k in kpt_names]

        if os.path.isfile(json_dir):
            self.json_files = [json_dir]
        else:
            self.json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

        if len(self.json_files) == 0:
            raise FileNotFoundError(f"目录 {json_dir} 下未找到任何 .json 文件！")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        orig_h, orig_w = data["imageHeight"], data["imageWidth"]

        img_path = os.path.join(os.path.dirname(json_path), data.get("imagePath", ""))
        if not os.path.exists(img_path):
            img_path = json_path.rsplit(".", 1)[0] + ".jpg"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # 按 group_id 分组
        groups = {}
        for shape in data.get("shapes", []):
            gid = shape.get("group_id", 0)
            if gid not in groups:
                groups[gid] = {"bbox": None, "mask_pts": None, "kpts": {}}

            label = shape.get("label", "").lower()
            stype = shape.get("shape_type", "")
            pts = shape.get("points", [])

            if stype == "rectangle":
                if len(pts) >= 2:
                    pts_arr = np.array(pts, dtype=np.float32)
                    x1 = float(pts_arr[:, 0].min())
                    y1 = float(pts_arr[:, 1].min())
                    x2 = float(pts_arr[:, 0].max())
                    y2 = float(pts_arr[:, 1].max())
                    groups[gid]["bbox"] = [x1, y1, x2, y2]
            elif stype == "polygon":
                groups[gid]["mask_pts"] = pts
            elif stype == "point" and label in self.kpt_names:
                if len(pts) >= 1:
                    groups[gid]["kpts"][label] = pts[0]

        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        cls_list, bbox_list, kpt_list, mask_list = [], [], [], []

        for gid, inst in sorted(groups.items()):
            if inst["bbox"] is None:
                continue

            x1, y1, x2, y2 = inst["bbox"]
            cx = ((x1 + x2) / 2.0) / orig_w
            cy = ((y1 + y2) / 2.0) / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            bbox_list.append([cx, cy, bw, bh])
            cls_list.append(0)

            # mask
            mask_mat = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            if inst["mask_pts"] is not None:
                poly_pts = np.array(inst["mask_pts"], dtype=np.float32)
                poly_pts[:, 0] *= scale_x
                poly_pts[:, 1] *= scale_y
                cv2.fillPoly(mask_mat, [poly_pts.astype(np.int32)], 1.0)
            mask_list.append(mask_mat)

            # keypoint
            kpt_arr = []
            for name in self.kpt_names:
                if name in inst["kpts"]:
                    kx, ky = inst["kpts"][name]
                    kpt_arr.append([kx / orig_w, ky / orig_h, 1.0])   # v=1 表示可见
                else:
                    kpt_arr.append([0.0, 0.0, 0.0])                    # v=0 表示缺失
            kpt_list.append(kpt_arr)

        if len(bbox_list) == 0:
            # 空图，返回一个空目标占位
            cls_t = torch.zeros((0, 1), dtype=torch.float32)
            bbox_t = torch.zeros((0, 4), dtype=torch.float32)
            kpt_t = torch.zeros((0, len(self.kpt_names), 3), dtype=torch.float32)
            mask_t = torch.zeros((0, self.img_size, self.img_size), dtype=torch.float32)
        else:
            cls_t = torch.tensor(cls_list, dtype=torch.float32).unsqueeze(1)
            bbox_t = torch.tensor(bbox_list, dtype=torch.float32)
            kpt_t = torch.tensor(kpt_list, dtype=torch.float32)
            mask_t = torch.tensor(np.array(mask_list), dtype=torch.float32)

        return {
            "img": img_tensor,
            "cls": cls_t,
            "bboxes": bbox_t,
            "keypoints": kpt_t,
            "masks": mask_t,
            "batch_idx": torch.zeros((cls_t.shape[0],), dtype=torch.long),
            "im_file": img_path,
            "ori_shape": (orig_h, orig_w),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
        }


def collate_fn(batch):
    """
    官方 loss 期望的是“拼在一起”的格式：
      batch["batch_idx"] 是 [total_N] 的全局索引
      batch["cls"]、batch["bboxes"]、batch["keypoints"]、batch["masks"] 是 [total_N, ...]
    """
    imgs = torch.stack([b["img"] for b in batch], dim=0)
    B = len(batch)

    cls_list, bbox_list, kpt_list, mask_list, bidx_list = [], [], [], [], []

    for b in range(B):
        n = batch[b]["cls"].shape[0]
        if n == 0:
            continue
        cls_list.append(batch[b]["cls"])
        bbox_list.append(batch[b]["bboxes"])
        kpt_list.append(batch[b]["keypoints"])
        mask_list.append(batch[b]["masks"])
        bidx_list.append(torch.full((n,), b, dtype=torch.long))

    if len(cls_list) == 0:
        cls_t = torch.zeros((0, 1), dtype=torch.float32)
        bbox_t = torch.zeros((0, 4), dtype=torch.float32)
        kpt_t = torch.zeros((0, len(batch[0]["keypoints"][0]) if batch[0]["keypoints"].numel() > 0 else 2, 3), dtype=torch.float32)
        mask_t = torch.zeros((0, batch[0]["masks"].shape[-2], batch[0]["masks"].shape[-1]), dtype=torch.float32)
        bidx_t = torch.zeros((0,), dtype=torch.long)
    else:
        cls_t = torch.cat(cls_list, dim=0)
        bbox_t = torch.cat(bbox_list, dim=0)
        kpt_t = torch.cat(kpt_list, dim=0)
        mask_t = torch.cat(mask_list, dim=0)
        bidx_t = torch.cat(bidx_list, dim=0)

    return {
        "img": imgs,
        "cls": cls_t,
        "bboxes": bbox_t,
        "keypoints": kpt_t,
        "masks": mask_t,
        "batch_idx": bidx_t,
    }