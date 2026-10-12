import copy
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Pose
from ultralytics.nn.modules.block import Proto
from ultralytics.nn.modules.conv import Conv


class PoseSegment(Pose):
    """
    继承官方 Pose，复用：
      - cv2：bbox 回归
      - cv3：分类
      - cv4：关键点
      - kpts_decode：关键点解码（anchor + stride + sigmoid）
      - decode_bboxes：DFL + dist2bbox

    新增：
      - cv5：分割 mask 系数分支
      - proto：mask 原型生成模块
    """

    def __init__(self, nc=1, kpt_shape=(2, 3), nm=32, npr=256,
                 reg_max=16, end2end=False, ch=()):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)

        self.nm = nm      # mask 系数数量
        self.npr = npr    # proto 数量

        # ---- 分割分支：仿照官方 Segment ----
        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c5, 3),
                Conv(c5, c5, 3),
                nn.Conv2d(c5, self.nm, 1),
            )
            for x in ch
        )
        self.proto = Proto(ch[0], self.npr, self.nm)

        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            mask_head=self.cv5,
        )

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            mask_head=self.one2one_cv5,
        )

    def forward_head(self, x, box_head, cls_head, pose_head, mask_head):
        # 先跑官方 Pose 的 forward_head，得到 boxes / scores / kpts
        preds = super().forward_head(x, box_head, cls_head, pose_head)

        # 再拼上分割 mask 系数
        if mask_head is not None:
            bs = x[0].shape[0]
            preds["mask_coefficient"] = torch.cat(
                [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
                dim=2,
            )
        return preds

    def forward(self, x):
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs

        # 如果 end2end，preds 是 {"one2many": {...}, "one2one": {...}}
        # 训练时只取 one2many，展开成平铺 dict
        if isinstance(preds, dict) and "one2many" in preds:
            preds = preds["one2many"]

        proto = self.proto(x[0])

        if isinstance(preds, dict):
            preds["proto"] = proto

        if self.training:
            return preds

        return (outputs, proto) if self.export else ((outputs[0], proto), preds)