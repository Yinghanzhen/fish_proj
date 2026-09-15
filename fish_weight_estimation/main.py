from fish_camera_class import TopFishData,SideFishData,CameraParams
from weight_estimation_model.calculate_length_height import calculate_heights,calculate_lengths
from ultralytics import YOLO
from weight_estimation_model.matching_algorithm import cross_view_fish_matching
from PCA_shape_prior.get_skeleton import extract_skeleton
from weight_estimation_model.get_3D_coordinate import process_fish_3d_data
from weight_estimation_model.weight_pretrained_model import estimate_fish_weight
from PCA_shape_prior.PCA_complete import batch_repair_fish_skeletons
from yolo_model.multi_task_yolo import FishPredictor
from data_handle.visualize import visualize_fishes_weight
import yaml
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO目标追踪工程化项目")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
         config = yaml.safe_load(f)

    #侧视图像输入专用YOLO
    model = YOLO("best.pt")
    side_results=model.predict(
        imgsz=640,
        device=0,
    )
    side_fishes = []

    #每条鱼实例化
    for i, (box, kpts) in enumerate(zip(side_results[0].boxes.xyxy, side_results[0].keypoints.xy)):
        x1, y1, x2, y2 = box.tolist()
        top_x, top_y = kpts[0].tolist()
        bottom_x, bottom_y = kpts[1].tolist()

        fish = SideFishData(
            fish_id=i,
            bbox=(x1, y1, x2, y2),
            top_kpt_2d=(top_x, top_y),
            bottom_kpt_2d=(bottom_x, bottom_y)
        )
        side_fishes.append(fish)

    #俯视图像输入改进过的结合注意力残差的多任务YOLO
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

    #获取二骨架线维坐标
    for fish in top_fishes:
            spine = extract_skeleton(fish.mask, fish.head_kpt_2d, fish.tail_kpt_2d)
            fish.spine_2d = spine

    #获取三维坐标         ！！！！！！！！！！！！！此步骤暂未完成，需按照实际相机情况调整！！！！！！！！！！！！！！
    top_camera_params = CameraParams(fx=1,fy=1,cx=1,cy=1,air_dist_mm=40)
    side_camera_params = CameraParams(fx=1,fy=1,cx=1,cy=1,air_dist_mm=40)
    process_fish_3d_data(top_fishes=top_fishes,
                         side_fishes=side_fishes,
                         side_cam_params=side_camera_params,
                         top_cam_params=top_camera_params)

    #鱼体遮挡补全
    batch_repair_fish_skeletons(top_fishes)

    #体长体高计算
    calculate_heights(side_fishes)
    calculate_lengths(top_fishes)

    #双视角鱼体匹配
    cross_view_fish_matching(top_fishes,side_fishes)

    #模型预测体重
    estimate_fish_weight(top_fishes,side_fishes)

    #可视化
    visualize_fishes_weight("top_view.jpg", top_fishes)


if __name__ == "__main__":
    main()