import os
from ultralytics import YOLO

def main():
    # ================== 配置 ==================
    DATA_YAML = r"D:\Project\fish_planB\data\yolo_seg\data.yaml"
    PROJECT = r"D:\Project\fish_planB\runs\segment"  # 输出目录
    EPOCHS = 200
    IMGSZ = 640
    BATCH = 2
    DEVICE = 0                            # GPU id
    WORKERS = 2

    model = YOLO(r"yolo26n-seg.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        pretrained="yolo26n-seg.pt",
        # ===== 关键参数 =====
        overlap_mask=False,    # ← 允许重叠像素被多条鱼共享
        mask_ratio=2,          # 掩码下采样比例，默认 4

        # 优化器参数
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,          # 旋转
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,            # 上下翻转（俯视图有用）
        fliplr=0.5,            # 左右翻转
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # 训练策略
        cos_lr=True,
        close_mosaic=10,
        save=True,
        save_period=-1,
        plots=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()