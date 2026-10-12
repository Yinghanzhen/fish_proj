import yaml
import argparse
from fish_camera_class import TopFishData,CameraParams
from ultralytics import YOLO
from PCA_shape_prior.get_skeleton import extract_skeleton
from weight_estimation_model.get_3D_coordinate import process_fish_3d_data
from PCA_shape_prior.PCA_complete import batch_repair_fish_skeletons
from data_handle.visualize import visualize_fishes_weight
from weight_estimation_model.top_fish_feature_extractor import process_top_fish_list
import cv2
def parse_args():
    parser = argparse.ArgumentParser(description="鱼体重量检测项目")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")

    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    img_path = r""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    H, W = img.shape[:2]
    fish_list=[]
    seg_model = YOLO(r"D:\Project\fish_planB\runs\segment\train2\weights\best.pt")
    results = seg_model(img,conf=0.35,iou=0.2)
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()  # [N, 4]
    masks = r.masks.data.cpu().numpy()  # [N, h, w] 下采样

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()

        # ---- mask 上采样到原图尺寸，并二值化 ----
        mask_small = masks[i]  # [h, w]
        mask_full = cv2.resize(
            mask_small, (W, H), interpolation=cv2.INTER_NEAREST
        ) > 0.5  # [H, W] bool
        fish = TopFishData(
            fish_id=i,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            head_kpt_2d=(0.0, 0.0),  # 后面用 pose 模型填
            tail_kpt_2d=(0.0, 0.0),
            mask=mask_full,  # 相对于整张图的 mask
        )

        fish_list.append(fish)

    for fish in fish_list:
        fish.spine_2d = extract_skeleton(fish)





if __name__ == '__main__':
    main()