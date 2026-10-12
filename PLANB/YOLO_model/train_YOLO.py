import glob
import json
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from multi_task_yolo import MultiTaskYOLO, MultiTaskYOLOLoss


class JSONMultiTaskDataset(Dataset):
    def __init__(self, json_dir, img_size=640, kpt_names=["head", "tail"]):
        super().__init__()
        self.img_size = img_size
        self.kpt_names = kpt_names

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

        groups = {}

        for shape in data.get("shapes", []):
            gid = shape.get("group_id")
            if gid is None:
                gid = 0

            if gid not in groups:
                groups[gid] = {"bbox": None, "mask_pts": None, "kpts": {}}

            label = shape.get("label", "")
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

            elif stype == "point":
                kpt_label = label.strip().lower()
                if kpt_label in [k.lower() for k in self.kpt_names]:
                    if len(pts) >= 1:
                        groups[gid]["kpts"][kpt_label] = pts[0]

        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        boxes_list, masks_list, kpts_list = [], [], []

        for gid, inst in sorted(groups.items()):
            if inst["bbox"] is None:
                continue

            x1, y1, x2, y2 = inst["bbox"]
            cx = ((x1 + x2) / 2.0) / orig_w
            cy = ((y1 + y2) / 2.0) / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            boxes_list.append([cx, cy, bw, bh])

            mask_mat = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            if inst["mask_pts"] is not None:
                poly_pts = np.array(inst["mask_pts"], dtype=np.float32)
                poly_pts[:, 0] *= scale_x
                poly_pts[:, 1] *= scale_y
                cv2.fillPoly(mask_mat, [poly_pts.astype(np.int32)], 1.0)
            masks_list.append(mask_mat)

            kpt_arr = []
            for kname in self.kpt_names:
                key = kname.lower()
                if key in inst["kpts"]:
                    kx, ky = inst["kpts"][key]
                    kpt_arr.extend([kx / orig_w, ky / orig_h, 1.0])
                else:
                    kpt_arr.extend([0.0, 0.0, 0.0])
            kpts_list.append(kpt_arr)

        if len(boxes_list) == 0:
            boxes_tensor = torch.zeros((1, 4), dtype=torch.float32)
            masks_tensor = torch.zeros(
                (1, self.img_size, self.img_size), dtype=torch.float32
            )
            kpts_tensor = torch.zeros(
                (1, len(self.kpt_names), 3), dtype=torch.float32
            )
        else:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
            masks_tensor = torch.tensor(np.array(masks_list), dtype=torch.float32)
            kpts_tensor = torch.tensor(kpts_list, dtype=torch.float32).view(
                -1, len(self.kpt_names), 3
            )

        return {
            "img": img_tensor,
            "boxes": boxes_tensor,
            "masks": masks_tensor,
            "keypoints": kpts_tensor,
        }


def multitask_collate_fn(batch):
    imgs = torch.stack([item["img"] for item in batch], dim=0)
    max_objs = max(item["boxes"].shape[0] for item in batch)

    batch_boxes, batch_masks, batch_kpts = [], [], []

    for item in batch:
        n_obj = item["boxes"].shape[0]
        pad_len = max_objs - n_obj

        if pad_len > 0:
            p_boxes = F.pad(item["boxes"], (0, 0, 0, pad_len))
            p_masks = F.pad(item["masks"], (0, 0, 0, 0, 0, pad_len))
            p_kpts = F.pad(item["keypoints"], (0, 0, 0, 0, 0, pad_len))
        else:
            p_boxes = item["boxes"]
            p_masks = item["masks"]
            p_kpts = item["keypoints"]

        batch_boxes.append(p_boxes)
        batch_masks.append(p_masks)
        batch_kpts.append(p_kpts)

    return {
        "img": imgs,
        "boxes": torch.stack(batch_boxes, dim=0),
        "masks": torch.stack(batch_masks, dim=0),
        "keypoints": torch.stack(batch_kpts, dim=0),
    }


def train():
    DATA_DIR = r"D:\Project\fish_planB\data\top_fish_data"
    KPT_NAMES = ["head", "tail"]
    CUSTOM_YOLO_PATH = r"D:\Project\fish_planB\config\yolo26_att_best.yaml"

    batch_size = 4
    epochs = 200
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"启动多任务 YOLO 训练，设备: {device}")

    full_dataset = JSONMultiTaskDataset(
        json_dir=DATA_DIR, img_size=640, kpt_names=KPT_NAMES
    )

    total_samples = len(full_dataset)
    print(f"实际读取 JSON 数量: {total_samples}")

    if total_samples < 1:
        raise RuntimeError("数据集为空")

    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multitask_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiTaskYOLO(
        backbone_model_path=CUSTOM_YOLO_PATH,
        num_classes=1,
        num_kpts=len(KPT_NAMES),
        target_indices=[16, 19, 22],
    ).to(device)

    criterion = MultiTaskYOLOLoss(
        w_box=7.5, w_seg=20.0, w_pose=12.0, w_obj=1.0,
        obj_pos_weight=50.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            imgs = batch["img"].to(device, non_blocking=True)

            targets = {
                "boxes": batch["boxes"].to(device, non_blocking=True),
                "masks": batch["masks"].to(device, non_blocking=True),
                "keypoints": batch["keypoints"].to(device, non_blocking=True),
            }

            optimizer.zero_grad(set_to_none=True)

            predictions = model(imgs)
            loss, loss_dict = criterion(predictions, targets)

            if not torch.isfinite(loss):
                print("检测到 NaN/Inf Loss，停止训练。")
                print(loss_dict)
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss += loss.item()

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] "
                f"| Loss: {loss_dict['total_loss']:.4f} "
                f"| Box: {loss_dict['loss_box']:.4f} "
                f"| Seg: {loss_dict['loss_seg']:.4f} "
                f"| Pose: {loss_dict['loss_pose']:.4f} "
                f"| Obj: {loss_dict['loss_obj']:.4f} "
                f"| Con: {loss_dict['loss_contrast']:.4f} "  # ← 加这一项
                f"| Pos: {loss_dict['positive_cells']:.0f}"
            )

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n>>> Epoch {epoch + 1}/{epochs} "
            f"| Train Loss: {avg_train_loss:.6f} "
            f"| LR: {current_lr:.7f}"
        )

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), "fish_multitask_best.pth")
            print(f"--> 保存最佳模型: Loss = {best_loss:.6f}")

    torch.save(model.state_dict(), "fish_multitask_last.pth")
    print("训练全部结束！")
    print(f"最佳 Train Loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()