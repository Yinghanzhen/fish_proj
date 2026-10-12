import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import make_anchors
from torchvision.ops import nms
import cv2


# ============================================================
# Proto 模块：Upsample + Conv2d，避免边界效应
# ============================================================
class ProtoModule(nn.Module):
    def __init__(self, in_channels=64, num_protos=32):
        super().__init__()
        self.proto = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_protos, kernel_size=1),
        )

    def forward(self, p3_feat):
        return self.proto(p3_feat)


# ============================================================
# MultiTaskHead：per-grid 输出
# ============================================================
class MultiTaskHead(nn.Module):
    def __init__(self, in_channels_list=(64, 128, 256), num_classes=1,
                 num_keypoints=2, num_protos=32, img_size=(640, 640)):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.num_protos = num_protos
        self.img_size = img_size
        self.nl = len(in_channels_list)

        self.register_buffer(
            "stride",
            torch.tensor([8.0, 16.0, 32.0]),
            persistent=False,
        )

        self.proto_module = ProtoModule(
            in_channels=in_channels_list[0], num_protos=num_protos
        )

        c2 = max(16, in_channels_list[0] // 4, 64)
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, c2, 3, padding=1),
                nn.BatchNorm2d(c2),
                nn.SiLU(),
                nn.Conv2d(c2, 4 + num_classes, 1),
            )
            for ch in in_channels_list
        ])

        c4 = max(in_channels_list[0] // 4, num_protos)
        self.segment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, c4, 3, padding=1),
                nn.BatchNorm2d(c4),
                nn.SiLU(),
                nn.Conv2d(c4, c4, 3, padding=1),
                nn.BatchNorm2d(c4),
                nn.SiLU(),
                nn.Conv2d(c4, num_protos, 1),
            )
            for ch in in_channels_list
        ])

        c5 = max(in_channels_list[0] // 4, num_keypoints * 3)
        self.pose_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, c5, 3, padding=1),
                nn.BatchNorm2d(c5),
                nn.SiLU(),
                nn.Conv2d(c5, num_keypoints * 3, 1),
            )
            for ch in in_channels_list
        ])

    def forward(self, feats):
        B = feats[0].shape[0]
        protos = self.proto_module(feats[0])

        box_list, coeff_list, kpt_list = [], [], []
        for i, feat in enumerate(feats):
            box_list.append(self.bbox_heads[i](feat).view(B, 4 + self.num_classes, -1))
            coeff_list.append(self.segment_heads[i](feat).view(B, self.num_protos, -1))
            kpt_list.append(self.pose_heads[i](feat).view(B, self.num_keypoints * 3, -1))

        boxes = torch.cat(box_list, dim=2)             # [B, 5, N]
        mask_coeffs = torch.cat(coeff_list, dim=2)     # [B, 32, N]
        keypoints = torch.cat(kpt_list, dim=2)         # [B, 6, N]

        anchors, strides = make_anchors(feats, self.stride, 0.5)

        return {
            "boxes": boxes,
            "mask_coeffs": mask_coeffs,
            "keypoints": keypoints,
            "protos": protos,
            "anchors": anchors,
            "strides": strides,
            "feats": feats,
        }

    @torch.no_grad()
    def decode(self, preds):
        anchors = preds["anchors"]
        boxes = preds["boxes"]

        bbox = torch.sigmoid(boxes[:, :4])
        cx, cy, bw, bh = bbox.unbind(1)
        x1 = (cx - bw / 2).clamp(0, 1)
        y1 = (cy - bh / 2).clamp(0, 1)
        x2 = (cx + bw / 2).clamp(0, 1)
        y2 = (cy + bh / 2).clamp(0, 1)
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        kpt_raw = preds["keypoints"]
        K = self.num_keypoints
        kpt_out = kpt_raw.clone()
        for k in range(K):
            dx = torch.sigmoid(kpt_raw[:, k * 3 + 0])
            dy = torch.sigmoid(kpt_raw[:, k * 3 + 1])
            v = kpt_raw[:, k * 3 + 2]
            kpt_out[:, k * 3 + 0] = dx.clamp(0, 1)
            kpt_out[:, k * 3 + 1] = dy.clamp(0, 1)
            kpt_out[:, k * 3 + 2] = v

        return boxes_xyxy, kpt_out


# ============================================================
# 多任务 YOLO
# ============================================================
class MultiTaskYOLO(nn.Module):
    def __init__(self, backbone_model_path, num_classes=1, num_kpts=2,
                 num_protos=32, target_indices=None):
        super().__init__()

        self.base_model = DetectionModel(cfg=backbone_model_path, nc=num_classes)

        layers = list(self.base_model.model.children())
        layers_count = len(layers)

        if target_indices is None:
            self.target_indices = [16, 19, 22]
        else:
            self.target_indices = target_indices

        if any(i >= layers_count for i in self.target_indices):
            raise IndexError(
                f"target_indices={self.target_indices} 超出模型层数 {layers_count}"
            )

        self.extracted_feats = []
        self._register_feature_hooks()

        dummy = torch.randn(1, 3, 640, 640)
        self.eval()
        with torch.no_grad():
            self._forward_backbone(dummy)
            in_channels_list = [f.shape[1] for f in self.extracted_feats]
            self.extracted_feats.clear()

        print(f"成功提取特征层索引 {self.target_indices}，通道: {in_channels_list}")

        self.head = MultiTaskHead(
            in_channels_list=in_channels_list,
            num_classes=num_classes,
            num_keypoints=num_kpts,
            num_protos=num_protos,
            img_size=(640, 640),
        )

    def _hook_fn(self, module, input, output):
        self.extracted_feats.append(output)

    def _register_feature_hooks(self):
        layers = list(self.base_model.model.children())
        for idx in self.target_indices:
            layers[idx].register_forward_hook(self._hook_fn)

    def _forward_backbone(self, x):
        self.extracted_feats.clear()
        _ = self.base_model(x)
        return self.extracted_feats

    def forward(self, x):
        feats = self._forward_backbone(x)
        return self.head(feats)


# ============================================================
# Loss：per-instance mask
# ============================================================
def bbox_ciou(pred_xyxy, gt_xyxy, eps=1e-7):
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    gw = (gx2 - gx1).clamp(min=0)
    gh = (gy2 - gy1).clamp(min=0)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = pw * ph + gw * gh - inter + eps
    iou = inter / union

    cw = torch.max(px2, gx2) - torch.min(px1, gx1)
    ch = torch.max(py2, gy2) - torch.min(py1, gy1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - (rho2 / c2 + alpha * v)


class MultiTaskYOLOLoss(nn.Module):
    def __init__(self, w_box=7.5, w_seg=20.0, w_pose=12.0, w_obj=1.0,
                 obj_pos_weight=50.0):
        super().__init__()
        self.w_box = w_box
        self.w_seg = w_seg
        self.w_pose = w_pose
        self.w_obj = w_obj

        self.bce_mask = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([2.0], dtype=torch.float32)
        )
        self.bce_obj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([obj_pos_weight], dtype=torch.float32)
        )

    @staticmethod
    def _cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _assign_with_obj(self, gt_boxes, anchors, strides, pred_obj, num_pos_max=9):
        """
        per-instance 分配：
          对每个 GT：
            1. 按面积选尺度
            2. 在该尺度内找离 GT 中心最近的 K 个网格
            3. 把这些网格分配给这个 GT
        """
        B, N, _ = gt_boxes.shape
        device = gt_boxes.device
        stride_vals = strides.squeeze(-1)
        unique_strides = torch.unique(stride_vals)

        pos_idx_list, pos_gt_idx_list = [], []

        for b in range(B):
            obj_b = pred_obj[b, 0]
            for n in range(N):
                cx, cy, bw, bh = gt_boxes[b, n]
                if bw <= 1e-6 or bh <= 1e-6:
                    continue

                area = float(bw * bh)
                if area < 0.02:
                    target_stride = unique_strides.min()
                elif area < 0.1:
                    target_stride = unique_strides[len(unique_strides) // 2]
                else:
                    target_stride = unique_strides.max()

                mask = (stride_vals == target_stride)
                cand_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                if cand_idx.numel() == 0:
                    continue

                cand_anchors = anchors[cand_idx]
                cand_strides = stride_vals[cand_idx]
                grid_size = 640.0 / cand_strides
                cx_px = cx * grid_size
                cy_px = cy * grid_size
                dist = (cand_anchors[:, 0] - cx_px) ** 2 + (cand_anchors[:, 1] - cy_px) ** 2
                radius_px = 0.15 * grid_size
                near_mask = dist < radius_px ** 2
                if near_mask.sum() == 0:
                    continue
                cand_idx = cand_idx[near_mask]
                cand_obj = obj_b[cand_idx]

                k = min(num_pos_max, cand_idx.numel())
                _, topk = cand_obj.topk(k, largest=True)
                chosen = cand_idx[topk]

                pos_idx_list.append(chosen)
                pos_gt_idx_list.append(
                    torch.full_like(chosen, n, dtype=torch.long)
                )

        if len(pos_idx_list) == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        pos_idx = torch.cat(pos_idx_list, dim=0)
        pos_gt_idx = torch.cat(pos_gt_idx_list, dim=0)
        return pos_idx, pos_gt_idx

    @staticmethod
    def instance_contrastive_loss(coeffs_pos, pos_gt_idx):
        """
        同一个 GT 的 coeffs 应该相似，不同 GT 的 coeffs 应该不相似。
        coeffs_pos: [B, 32, M]
        pos_gt_idx: [M]
        """
        # 归一化
        coeffs_norm = F.normalize(coeffs_pos, dim=1)  # [B, 32, M]

        # 两两相似度
        sim = torch.einsum("bcm,bcn->bmn", coeffs_norm, coeffs_norm)  # [B, M, M]

        # 同一个 GT 的 mask 应该相似（> 0.9）
        # 不同 GT 的 mask 应该不相似（< 0.1）
        same_gt = (pos_gt_idx[:, None] == pos_gt_idx[None, :]).float()
        same_gt = same_gt * 2.0 - 1.0

        loss = ((sim - same_gt) ** 2).mean()
        return loss

    @staticmethod
    def prototype_diversity_loss(protos):
        """
        不同的 proto 应该不相似。
        protos: [B, 32, H, W]
        """
        B, C, H, W = protos.shape
        # 把每个 proto 展平
        protos_flat = protos.view(B, C, -1)  # [B, 32, H*W]
        protos_norm = F.normalize(protos_flat, dim=-1)  # [B, 32, H*W]

        # 两两相似度
        sim = torch.einsum("bch,bdh->bcd", protos_norm, protos_norm)  # [B, 32, 32]

        # 对角线应该是 1，非对角线应该接近 0
        eye = torch.eye(C, device=protos.device).unsqueeze(0)  # [1, 32, 32]
        loss = ((sim - eye) ** 2).mean()
        return loss

    def forward(self, preds, targets):
        device = preds["boxes"].device
        anchors = preds["anchors"]
        strides = preds["strides"]
        N = anchors.shape[0]

        boxes = preds["boxes"]
        coeffs = preds["mask_coeffs"]
        kpts = preds["keypoints"]
        protos = preds["protos"]

        gt_boxes = targets["boxes"].to(device)
        gt_kpts = targets["keypoints"].to(device)
        gt_masks = targets["masks"].to(device)

        # ---- 1. 正样本分配 ----
        pred_obj_for_assign = boxes[:, 4:5]
        pos_idx, pos_gt_idx = self._assign_with_obj(
            gt_boxes, anchors, strides, pred_obj_for_assign, num_pos_max=9
        )

        target_obj = torch.zeros((boxes.shape[0], 1, N), device=device)
        if pos_idx.numel() > 0:
            target_obj[:, 0, pos_idx] = 1.0

        # ---- 2. bbox loss ----
        boxes_xyxy, _ = self._decode_for_loss(boxes, anchors)

        loss_box = torch.tensor(0.0, device=device)
        loss_pose = torch.tensor(0.0, device=device)
        loss_seg = torch.tensor(0.0, device=device)
        loss_contrast = torch.tensor(0.0, device=device)

        if pos_idx.numel() > 0:
            pred_box_pos = boxes_xyxy[:, :, pos_idx].permute(0, 2, 1)
            gt_box_pos_xyxy = self._cxcywh_to_xyxy(gt_boxes[:, pos_gt_idx, :])
            iou = bbox_ciou(pred_box_pos, gt_box_pos_xyxy)
            loss_box = (1 - iou).mean()

            # ---- keypoint ----
            pred_kpt_pos = kpts[:, :, pos_idx].permute(0, 2, 1)
            pred_kpt_pos = pred_kpt_pos.clone()
            pred_kpt_pos[..., 0::3] = torch.sigmoid(pred_kpt_pos[..., 0::3])
            pred_kpt_pos[..., 1::3] = torch.sigmoid(pred_kpt_pos[..., 1::3])

            gt_kpt_pos = gt_kpts[:, pos_gt_idx, :, :].reshape(
                gt_kpts.shape[0], -1, self._kpt_dim(gt_kpts)
            )

            gt_vis = gt_kpts[:, pos_gt_idx, :, 2]
            kpt_mask = (gt_vis > 0).float().unsqueeze(-1).repeat(1, 1, 1, 3)
            kpt_mask = kpt_mask.reshape(gt_kpt_pos.shape[0], gt_kpt_pos.shape[1], -1)

            loss_pose = F.mse_loss(
                pred_kpt_pos * kpt_mask, gt_kpt_pos * kpt_mask, reduction="sum"
            ) / kpt_mask.sum().clamp(min=1.0)

            # ---- mask：per-instance ----
            coeffs_pos = coeffs[:, :, pos_idx]                # [B, 32, M]
            pred_masks = torch.einsum(
                "bmc,bchw->bmhw", coeffs_pos.permute(0, 2, 1), protos
            )
            gt_mask_pos = gt_masks[:, pos_gt_idx]             # [B, M, 640, 640]
            B, M, _, _ = pred_masks.shape
            pred_masks_up = F.interpolate(
                pred_masks.view(B * M, 1, *pred_masks.shape[-2:]),
                size=gt_mask_pos.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).view(B, M, *gt_mask_pos.shape[-2:])

            loss_seg = self.bce_mask(pred_masks_up, gt_mask_pos)
            loss_seg_reg = 0.1 * (coeffs_pos ** 2).mean() + 0.1 * (protos ** 2).mean()
            loss_seg = loss_seg + 0.01 * loss_seg_reg

            # ===== 新增：实例对比 loss =====
            loss_contrast = self.instance_contrastive_loss(coeffs_pos, pos_gt_idx)

        # ---- 3. objectness loss ----
        pred_obj = boxes[:, 4:5]
        loss_obj = self.bce_obj(pred_obj, target_obj)

        total_loss = (
                self.w_box * loss_box
                + self.w_seg * loss_seg
                + self.w_pose * loss_pose
                + self.w_obj * loss_obj
                + 20.0 * loss_contrast  # ← 加这一行
                + 5.0* self.prototype_diversity_loss(protos)  # ← 加这一行
        )

        return total_loss, {
            "loss_box": float(loss_box.item()),
            "loss_seg": float(loss_seg.item()),
            "loss_pose": float(loss_pose.item()),
            "loss_obj": float(loss_obj.item()),
            "loss_contrast": float(loss_contrast.item()),  # ← 加这一行
            "total_loss": float(total_loss.item()),
            "positive_cells": float(target_obj.sum().item()),
        }

    @staticmethod
    def _kpt_dim(gt_kpts):
        return gt_kpts.shape[2] * gt_kpts.shape[3]

    @staticmethod
    def _decode_for_loss(boxes, anchors):
        bbox = torch.sigmoid(boxes[:, :4])
        cx, cy, bw, bh = bbox.unbind(1)
        x1 = (cx - bw / 2).clamp(0, 1)
        y1 = (cy - bh / 2).clamp(0, 1)
        x2 = (cx + bw / 2).clamp(0, 1)
        y2 = (cy + bh / 2).clamp(0, 1)
        return torch.stack([x1, y1, x2, y2], dim=1), anchors


# ============================================================
# 推理器：多鱼实例分割
# ============================================================
class FishPredictor:
    def __init__(self, model_path, backbone_model_path, device="cuda"):
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.model = MultiTaskYOLO(
            backbone_model_path=backbone_model_path,
            num_classes=1,
            num_kpts=2,
            target_indices=[16, 19, 22],
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, img_path, conf_threshold=0.25, iou_threshold=0.5,
                max_det=100, pool_kernel=3):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640))
        tensor = (
            torch.from_numpy(resized).permute(2, 0, 1).float()
            .unsqueeze(0).to(self.device) / 255.0
        )

        preds = self.model(tensor)
        boxes_xyxy, kpts = self.model.head.decode(preds)
        obj = torch.sigmoid(preds["boxes"][:, 4])          # [B, N]
        coeffs = preds["mask_coeffs"]                      # [B, 32, N]
        protos = preds["protos"]                           # [B, 32, 160, 160]
        feats = preds["feats"]

        # 计算每个网格的 (H, W)
        H, W = feats[0].shape[2], feats[0].shape[3]
        feat_sizes = [(f.shape[2], f.shape[3]) for f in feats]

        B = obj.shape[0]
        results = []

        for b in range(B):
            scores = obj[b]                                # [N]

            # ===== 1. 找 obj 的局部极大值 =====
            # 把 [N] 重新组织成每个尺度的 [H_i * W_i]
            peaks_all = []
            offset = 0
            for (H_i, W_i) in feat_sizes:
                n_i = H_i * W_i
                score_i = scores[offset:offset + n_i].view(1, 1, H_i, W_i)
                obj_max_i = F.max_pool2d(
                    score_i, pool_kernel, stride=1, padding=pool_kernel // 2
                )
                is_peak_i = (score_i == obj_max_i).view(-1)
                peaks_all.append(is_peak_i)
                offset += n_i

            peaks = torch.cat(peaks_all, dim=0)            # [N]
            peaks = peaks & (scores > conf_threshold)
            peak_idx = torch.nonzero(peaks, as_tuple=False).squeeze(1)

            if len(peak_idx) == 0:
                results.append([])
                continue

            # ===== 2. NMS 去重 =====
            boxes_kept = boxes_xyxy[b, :, peak_idx].t()    # [K, 4]
            boxes_kept = boxes_kept.clamp(0, 1)
            scores_kept = scores[peak_idx]
            keep_nms = nms(boxes_kept, scores_kept, iou_threshold)
            keep_nms = keep_nms[:max_det]
            peak_idx = peak_idx[keep_nms]
            scores_kept = scores_kept[keep_nms]

            # ===== 3. 每个峰值输出自己的 mask / bbox / keypoint =====
            per_img = []
            for pid in peak_idx:
                coeff = coeffs[b, :, pid]                  # [32]
                mask = torch.einsum("c,chw->hw", coeff, protos[b])
                mask = F.interpolate(
                    mask[None, None], size=(h, w),
                    mode="bilinear", align_corners=False,
                )[0, 0]
                mask = torch.sigmoid(mask) > 0.5

                box = boxes_xyxy[b, :, pid].clamp(0, 1).tolist()
                x1, y1, x2, y2 = box
                x1 *= w; x2 *= w; y1 *= h; y2 *= h

                kpt = kpts[b, :, pid].view(2, 3)           # [2, 3]
                head = kpt[0]
                tail = kpt[1]

                per_img.append({
                    "bbox": (x1, y1, x2, y2),
                    "head": (head[0].item() * w, head[1].item() * h, head[2].item()),
                    "tail": (tail[0].item() * w, tail[1].item() * h, tail[2].item()),
                    "mask": mask.cpu().numpy(),
                    "confidence": float(scores_kept[keep_nms == pid].item())
                        if (keep_nms == pid).any() else float(scores[pid].item()),
                })

            results.append(per_img)

        return results[0] if B == 1 else results