import os
from ultralytics import YOLO


def main():
    # ================== 配置 ==================
    DATA_YAML = r"D:\Project\fish_planB\data\yolo_pose\data_pose.yaml"
    PRETRAINED = "yolov26n-pose.pt"
    PROJECT = r"D:\Project\fish_planB\runs\pose"
    NAME = "fish_pose"
    EPOCHS = 200
    IMGSZ = 640
    BATCH = 3
    DEVICE = 0
    WORKERS = 2

    model = YOLO(PRETRAINED)

    # ===== 关键：修改 kpt_shape 从 (17, 3) 改成 (2, 3) =====
    model.model.yaml["kpt_shape"] = [2, 3]
    # 重建 head，让关键点输出通道从 51 变成 6
    from ultralytics.nn.modules.head import Pose
    # 获取 Detect head 的输入通道
    head = model.model.model[-1]
    ch = [head.cv2[i][0].conv.in_channels for i in range(head.nl)]

    # 用新的 kpt_shape 重建 Pose head
    model.model.model[-1] = Pose(
        nc=1,
        kpt_shape=(2, 3),
        reg_max=getattr(head, "reg_max", 16),
        end2end=getattr(head, "_end2end", False),
        ch=tuple(ch),
    )
    model.model.model[-1].stride = head.stride
    model.model.model[-1].f = head.f
    model.model.model[-1].i = head.i
    model.model.model[-1].type = "pose"

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,

        # 优化器
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,

        # 训练策略
        cos_lr=True,
        close_mosaic=10,
        patience=50,
        save=True,
        save_period=-1,
        plots=True,
        val=True,
        verbose=True,
    )

    print(f"最佳权重: {os.path.join(PROJECT, NAME, 'weights', 'best.pt')}")

if __name__ == "__main__":
    main()