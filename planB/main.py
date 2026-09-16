import yaml
import argparse
from YOLO_model.multi_task_yolo import FishPredictor
from fish_camera_class import TopFishData,CameraParams
from weight_estimation_model.calculate_length import calculate_lengths
from ultralytics import YOLO
from PCA_shape_prior.get_skeleton import extract_skeleton
from weight_estimation_model.get_3D_coordinate import process_fish_3d_data
from PCA_shape_prior.PCA_complete import batch_repair_fish_skeletons
from data_handle.visualize import visualize_fishes_weight
from weight_estimation_model.top_fish_feature_extractor import process_top_fish_list
def parse_args():
    parser = argparse.ArgumentParser(description="鱼体重量检测项目")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")

    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    predictor = FishPredictor("fish_multitask_best.pth")
    top_results = predictor.predict("top_view.jpg")
    top_fishes = [
        TopFishData(
            fish_id=i,
            bbox=res["bbox"],
            head_kpt_2d=res["head"],
            tail_kpt_2d=res["tail"],
            mask=res["mask"],
        )
        for i, res in enumerate(top_results)
    ]
    # 获取二骨架线维坐标
    for fish in top_fishes:
        spine = extract_skeleton(fish.mask, fish.head_kpt_2d, fish.tail_kpt_2d)
        fish.spine_2d = spine

    # 获取三维坐标         ！！！！！！！！！！！！！此步骤暂未完成，需按照实际相机情况调整！！！！！！！！！！！！！！
    top_camera_params = CameraParams(fx=1, fy=1, cx=1, cy=1, air_dist_mm=40)
    process_fish_3d_data(top_fishes=top_fishes,
                         top_cam_params=top_camera_params)

    # 鱼体遮挡补全
    batch_repair_fish_skeletons(top_fishes)
    calculate_lengths(top_fishes)

    process_top_fish_list(top_fishes)




    visualize_fishes_weight("top_view.jpg", top_fishes)

if __name__ == '__main__':
    main()