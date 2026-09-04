import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import cv2

class ProtoModule(nn.Module):
    """原型掩码生成模块 (80x80 -> 160x160)"""

    def __init__(self, in_channels=128, num_protos=32):
        super().__init__()
        self.proto = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_protos, kernel_size=1),
        )

    def forward(self, p3_feat):
        return self.proto(p3_feat)


class MultiTaskHead(nn.Module):
    """YOLO 多任务 Head：预测 Bbox、Keypoints 与 Mask"""

    def __init__(
        self,
        in_channels_list=[128, 256, 512],
        num_classes=1,
        num_keypoints=2,
        num_protos=32,
        img_size=(640, 640),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.num_protos = num_protos
        self.img_size = img_size

        self.proto_module = ProtoModule(
            in_channels=in_channels_list[0], num_protos=num_protos
        )

        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch),
                nn.BatchNorm2d(ch),
                nn.SiLU(),
                nn.Conv2d(ch, 4 + num_classes, kernel_size=1),
            )
            for ch in in_channels_list
        ])

        self.segment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(ch, num_protos, kernel_size=1),
            )
            for ch in in_channels_list
        ])

        self.pose_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(ch, num_keypoints * 3, kernel_size=1),
            )
            for ch in in_channels_list
        ])

    def forward(self, feats):
        B = feats[0].shape[0]
        protos = self.proto_module(feats[0])  # [B, 32, 160, 160]

        bbox_outputs, mask_coeff_list, pose_outputs = [], [], []

        for i, feat in enumerate(feats):
            bbox_outputs.append(self.bbox_heads[i](feat))
            pose_outputs.append(self.pose_heads[i](feat))
            s_out = self.segment_heads[i](feat)
            mask_coeff_list.append(s_out.flatten(2))

        # 合成低分辨率 Mask [B, 8400, 160, 160]
        mask_coeffs = torch.cat(mask_coeff_list, dim=2).transpose(
            1, 2
        )  # [B, 8400, 32]
        protos_flat = protos.view(B, self.num_protos, -1)  # [B, 32, 25600]
        raw_masks_160 = torch.bmm(mask_coeffs, protos_flat).view(
            B, -1, 160, 160
        )

        if not self.training:
            masks_640 = F.interpolate(
                raw_masks_160,
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            masks_out = torch.sigmoid(masks_640)
        else:
            masks_out = raw_masks_160

        return {
            "boxes": bbox_outputs,
            "masks_raw": masks_out,
            "keypoints": pose_outputs,
            "protos": protos,
        }

class MultiTaskYOLO(nn.Module):
    """完整多任务网络"""

    def __init__(
        self,
        backbone_model_path="yolo11n.pt",
        num_classes=1,
        num_kpts=2,
        num_protos=32,
    ):
        super().__init__()
        try:
            self.base_model = YOLO(backbone_model_path).model
        except Exception:
            self.base_model = YOLO("yolo11n.pt").model

        self.extracted_feats = []
        self._register_feature_hooks()

        dummy_input = torch.randn(1, 3, 640, 640)
        self.eval()
        with torch.no_grad():
            self._forward_backbone(dummy_input)
            in_channels_list = [f.shape[1] for f in self.extracted_feats]
            self.extracted_feats.clear()

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
        target_indices = [15, 18, 21] if len(layers) > 22 else [-4, -3, -2]
        for idx in target_indices:
            layers[idx].register_forward_hook(self._hook_fn)

    def _forward_backbone(self, x):
        self.extracted_feats.clear()
        for layer in list(self.base_model.model.children())[:-1]:
            x = layer(x)
        return self.extracted_feats

    def forward(self, x):
        feats = self._forward_backbone(x)
        return self.head(feats)


class MultiTaskYOLOLoss(nn.Module):
    """多任务联合 Loss 函数"""
    def __init__(self, w_box=7.5, w_seg=2.5, w_pose=12.0):
        super().__init__()
        self.w_box = w_box
        self.w_seg = w_seg
        self.w_pose = w_pose

        self.bce_mask = nn.BCEWithLogitsLoss()
        self.mse_kpt = nn.MSELoss()
        self.l1_box = nn.L1Loss()

    def forward(self, preds, targets):
        device = preds["protos"].device

        # 1. Bbox Loss (以 P3 特征图计算示范)
        pred_boxes_p3 = preds["boxes"][0]
        gt_boxes = targets["boxes"].to(device)
        loss_box = self.l1_box(
            pred_boxes_p3[:, :4, :10, :10],
            gt_boxes[:, :1, :4].expand(-1, -1, 10, 10),
        )

        # 2. Mask Loss (160x160 降采样分辨率对比)
        gt_masks = targets["masks"].to(device)
        gt_masks_160 = F.interpolate(gt_masks, size=(160, 160), mode="nearest")
        raw_masks_160 = preds["masks_raw"]
        loss_seg = self.bce_mask(raw_masks_160[:, :1, :, :], gt_masks_160)

        # 3. Keypoints Loss
        pred_kpts_p3 = preds["keypoints"][0]
        gt_kpts = targets["keypoints"].to(device)
        loss_pose = self.mse_kpt(
            pred_kpts_p3[:, : gt_kpts.shape[2] * 3, :1, :1].view(
                -1, gt_kpts.shape[1], gt_kpts.shape[2], 3
            ),
            gt_kpts,
        )

        total_loss = (
            (self.w_box * loss_box)
            + (self.w_seg * loss_seg)
            + (self.w_pose * loss_pose)
        )

        return total_loss, {
            "loss_box": loss_box.item(),
            "loss_seg": loss_seg.item(),
            "loss_pose": loss_pose.item(),
            "total_loss": total_loss.item(),
        }


class FishPredictor:

    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model = MultiTaskYOLO(num_classes=1, num_kpts=2)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device).eval()

    def predict(self, img_path):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        tensor = (
            torch.from_numpy(
                cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
            )
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0
        ).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)

        # 解析概率 Mask 还原回原图大小
        mask = (
            (
                torch.sigmoid(
                    F.interpolate(
                        preds["masks_raw"], size=(h, w), mode="bilinear"
                    )
                )[0, 0]
                > 0.5
            )
            .cpu()
            .numpy()
        )

        # 这里假设返回格式：(Bbox, Head_kpt, Tail_kpt, Mask)
        # 实际使用中可以根据模型输出精细解码坐标
        return [{
            "bbox": (0.0, 0.0, float(w), float(h)),
            "head": (w * 0.4, h * 0.5),
            "tail": (w * 0.6, h * 0.5),
            "mask": mask,
        }]