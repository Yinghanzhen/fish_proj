import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import cv2


class ProtoModule(nn.Module):
    """原型掩码生成模块 (P3: 80x80 -> 160x160)"""

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
    """YOLO 多任务 Head：预测 Bbox、Keypoints 与 Prototype Mask"""

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
            s_out = self.segment_heads[i](feat)  # [B, 32, H_i, W_i]
            mask_coeff_list.append(s_out.flatten(2))

        # 拼接多尺度掩码系数 [B, 32, N_anchors] -> [B, N_anchors, 32]
        mask_coeffs = torch.cat(mask_coeff_list, dim=2).transpose(1, 2)

        # 针对简单单目标或全局掩码：取系数均值生成图像级特征图合成
        # [B, 32] -> [B, 32, 1, 1]
        global_mask_coeff = mask_coeffs.mean(dim=1).view(B, self.num_protos, 1, 1)
        raw_masks_160 = (protos * global_mask_coeff).sum(dim=1, keepdim=True)  # [B, 1, 160, 160]

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
    """支持自定义/注册版 YOLO Backbone 的多任务网络"""

    def __init__(
        self,
        backbone_model_path="yolo11n.pt",
        num_classes=1,
        num_kpts=2,
        num_protos=32,
        target_indices=None,
    ):
        super().__init__()

        # 1. 加载包含自定义注册模块的模型
        try:
            self.yolo_wrapper = YOLO(backbone_model_path)
            self.base_model = self.yolo_wrapper.model
        except Exception as e:
            print(f"加载模型 {backbone_model_path} 失败，请检查注册表或路径: {e}")
            raise e

        # 2. 自动匹配层索引
        layers = list(self.base_model.model.children())
        layers_count = len(layers)

        if target_indices is None:
            # 默认提取 Backbone 输出的 P3, P4, P5
            self.target_indices = [
                layers_count - 4,
                layers_count - 3,
                layers_count - 2,
            ]
        else:
            self.target_indices = target_indices

        self.extracted_feats = []
        self._register_feature_hooks()

        # 3. 动态获取通道数
        dummy_input = torch.randn(1, 3, 640, 640)
        self.eval()
        with torch.no_grad():
            self._forward_backbone(dummy_input)
            in_channels_list = [f.shape[1] for f in self.extracted_feats]
            self.extracted_feats.clear()

        print(
            f"成功提取特征层索引 {self.target_indices}，对应通道维度: {in_channels_list}"
        )

        # 4. 初始化多任务 Head
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
        self.extracted_feats.clear()  # 必须清空保证不累加
        # 调用 base_model 完整正向拓扑图，触发 Forward Hook
        _ = self.base_model(x)
        return self.extracted_feats

    def forward(self, x):
        feats = self._forward_backbone(x)
        return self.head(feats)


class MultiTaskYOLOLoss(nn.Module):
    """多任务联合 Loss 函数"""

    def __init__(self, w_box=7.5, w_seg=2.5, w_pose=12.0, w_obj=1.0):
        super().__init__()
        self.w_box = w_box
        self.w_seg = w_seg
        self.w_pose = w_pose
        self.w_obj = w_obj

        self.bce_mask = nn.BCEWithLogitsLoss()
        self.bce_obj = nn.BCEWithLogitsLoss()
        self.mse_kpt = nn.MSELoss()
        self.l1_box = nn.L1Loss()

    def forward(self, preds, targets):
        device = preds["protos"].device

        # 1. Bbox 坐标 Loss 与 目标置信度 Loss (基于 P3 特征图示例)
        pred_p3 = preds["boxes"][0]  # [B, 5, H, W]
        pred_boxes_p3 = pred_p3[:, :4, :, :]
        pred_obj_p3 = pred_p3[:, 4:5, :, :]

        gt_boxes = targets["boxes"].to(device)  # [B, N, 4]

        # 动态对齐真实框均值做示例计算
        gt_box_center = gt_boxes.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 4, 1]
        target_box_map = gt_box_center.expand(-1, -1, -1, pred_boxes_p3.shape[-1]).permute(0, 2, 3, 1)
        target_box_map = F.interpolate(target_box_map, size=pred_boxes_p3.shape[2:])

        loss_box = self.l1_box(pred_boxes_p3, target_box_map)

        gt_obj = torch.ones_like(pred_obj_p3)
        loss_obj = self.bce_obj(pred_obj_p3, gt_obj)

        # 2. Mask Loss (计算全局单目标掩码)
        gt_masks = targets["masks"].to(device)  # [B, N, 640, 640] 或 [B, N, 160, 160]
        if gt_masks.ndim == 3:
            gt_masks = gt_masks.unsqueeze(1)
        elif gt_masks.ndim == 4 and gt_masks.shape[1] > 1:
            gt_masks = gt_masks.max(dim=1, keepdim=True)[0]  # 融合同一个图内的多目标 Mask

        gt_masks_160 = F.interpolate(
            gt_masks, size=(160, 160), mode="nearest"
        )
        raw_masks_160 = preds["masks_raw"]  # [B, 1, 160, 160]
        loss_seg = self.bce_mask(raw_masks_160, gt_masks_160)

        # 3. Keypoints Loss
        pred_kpt_p3 = preds["keypoints"][0]  # [B, num_kpts*3, H, W]
        gt_kpts = targets["keypoints"].to(device)  # [B, N, num_kpts, 3]

        # 提取全局均值 keypoints 进行回归
        gt_kpts_mean = gt_kpts.mean(dim=1)  # [B, num_kpts, 3]
        pred_kpt_flat = pred_kpt_p3.mean(dim=[-2, -1]).view(-1, gt_kpts.shape[2], 3)
        loss_pose = self.mse_kpt(pred_kpt_flat, gt_kpts_mean)

        total_loss = (
            (self.w_box * loss_box)
            + (self.w_seg * loss_seg)
            + (self.w_pose * loss_pose)
            + (self.w_obj * loss_obj)
        )

        return total_loss, {
            "loss_box": loss_box.item(),
            "loss_seg": loss_seg.item(),
            "loss_pose": loss_pose.item(),
            "loss_obj": loss_obj.item(),
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