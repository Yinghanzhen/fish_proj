import os
import matplotlib.pyplot as plt
from multi_task_yolo import FishPredictor

BACKBONE_PATH = r"D:\Project\fish_planB\config\yolo26_att_best.yaml"
MODEL_PATH = "fish_multitask_best.pth"
IMAGE_PATH = r"D:\Project\fish_planB\data\top_fish_data\5807.jpg"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型权重: {MODEL_PATH}")

predictor = FishPredictor(
    model_path=MODEL_PATH,
    backbone_model_path=BACKBONE_PATH,
    device="cuda",
)

results = predictor.predict(
    IMAGE_PATH,
    conf_threshold=0.25,
    iou_threshold=0.5,
    max_det=100,
    pool_kernel=3,
)

print(f"检测到 {len(results)} 条鱼")
for i, r in enumerate(results):
    print(f"\n第 {i + 1} 条鱼:")
    print("BBox:", r["bbox"])
    print("Head:", r["head"])
    print("Tail:", r["tail"])
    print("Confidence:", r["confidence"])
    print("Mask 非零像素:", int(r["mask"].sum()))

    plt.figure()
    plt.imshow(r["mask"], cmap="gray")
    plt.title(f"Fish {i+1} mask")
    plt.savefig(f"mask_debug_{i+1}.png")
    plt.close()
    print(f"mask 已保存: mask_debug_{i+1}.png")