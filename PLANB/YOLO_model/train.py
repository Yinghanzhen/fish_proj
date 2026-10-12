import os
import torch
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.tal import TaskAlignedAssigner

from multi_yolo_v2 import PoseSegment
from json_dataset import JSONPoseSegDataset, collate_fn


# ============================================================
# 用你的 yaml 构建 DetectionModel，然后把最后的 Detect 换成 PoseSegment
# ============================================================
def build_model(yaml_path, nc=1, kpt_shape=(2, 3), nm=32, npr=256):
    model = DetectionModel(cfg=yaml_path, nc=nc)

    old_head = model.model[-1]
    ch = [old_head.cv2[i][0].conv.in_channels for i in range(old_head.nl)]
    reg_max = getattr(old_head, "reg_max", 16)
    end2end = getattr(old_head, "end2end", False)

    print(f"替换 Detect 为 PoseSegment，输入通道: {ch}")

    new_head = PoseSegment(
        nc=nc, kpt_shape=kpt_shape, nm=nm, npr=npr,
        reg_max=reg_max, end2end=end2end, ch=tuple(ch),
    )

    # ===== 抄旧 head 的属性，_predict_once 需要 =====
    new_head.f = old_head.f
    new_head.i = old_head.i
    new_head.type = old_head.type
    new_head.stride = old_head.stride

    model.model[-1] = new_head

    # ===== 补官方 loss 需要的属性 =====
    class DummyArgs:
        def __init__(self, nc):
            self.nc = nc
            self.box = 7.5
            self.cls = 0.5
            self.dfl = 1.5
            self.pose = 12.0
            self.kobj = 1.0
            self.label_smoothing = 0.0
            self.overlap_mask = True
            self.mask_ratio = 4
            self.nbs = 64
            self.tal_topk = 10

    model.args = DummyArgs(nc)
    model.nc = nc
    model.reg_max = reg_max
    model.stride = old_head.stride

    return model


# ============================================================
# 训练
# ============================================================
def train():
    DATA_DIR = r"D:\Project\fish_planB\data\top_fish_data"
    YAML_PATH = r"D:\Project\fish_planB\config\yolo26_att_best.yaml"

    KPT_NAMES = ["head", "tail"]
    KPT_SHAPE = (2, 3)
    NC = 1
    NM = 32
    NPR = 256

    BATCH = 3
    EPOCHS = 77
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"设备: {DEVICE}")

    # ---- 数据 ----
    dataset = JSONPoseSegDataset(DATA_DIR, img_size=640, kpt_names=KPT_NAMES)
    loader = DataLoader(
        dataset, batch_size=BATCH, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- 模型 ----
    model = build_model(YAML_PATH, nc=NC, kpt_shape=KPT_SHAPE, nm=NM, npr=NPR)
    model.to(DEVICE).train()

    # ---- 官方 loss ----
    # 三个任务分别用官方 loss，再手工加权
    det_loss_fn = v8DetectionLoss(model)
    pose_loss_fn = v8PoseLoss(model)
    seg_loss_fn = v8SegmentationLoss(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            imgs = batch["img"].to(DEVICE, non_blocking=True)
            batch_dev = {
                "batch_idx": batch["batch_idx"].to(DEVICE),
                "cls": batch["cls"].to(DEVICE),
                "bboxes": batch["bboxes"].to(DEVICE),
                "keypoints": batch["keypoints"].to(DEVICE),
                "masks": batch["masks"].to(DEVICE),
                "img": imgs,
            }

            optimizer.zero_grad(set_to_none=True)

            # 前向：返回 dict，包含 boxes / scores / kpts / mask_coefficient / proto / feats
            preds = model(imgs)

            # ---- 官方 loss ----
            # 注意：官方 loss 的 forward 签名是 (preds, batch)
            loss_det, det_items = det_loss_fn(preds, batch_dev)
            loss_pose, pose_items = pose_loss_fn(preds, batch_dev)
            loss_seg, seg_items = seg_loss_fn(preds, batch_dev)

            loss = loss_det.sum() + loss_pose.sum() + loss_seg.sum()

            if not torch.isfinite(loss):
                print("检测到 NaN/Inf，停止训练。")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] Step [{step + 1}/{len(loader)}] "
                f"| Det: {loss_det.sum().item():.4f} | Pose: {loss_pose.sum().item():.4f} "
                f"| Seg: {loss_seg.sum().item():.4f} | Total: {loss.item():.4f}"
            )

        scheduler.step()
        avg = total_loss / max(len(loader), 1)
        print(f"\n>>> Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg:.6f} | LR: {optimizer.param_groups[0]['lr']:.7f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "fish_multitask_best.pth")
            print(f"--> 保存最佳模型: Loss = {best_loss:.6f}")

    torch.save(model.state_dict(), "fish_multitask_last.pth")
    print(f"训练结束，最佳 Loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()