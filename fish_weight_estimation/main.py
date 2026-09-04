from fish_class import TopFishData,SideFishData
from weight_estimation_model.calculate_length_height import calculate_heights,calculate_lengths
from ultralytics import YOLO
from weight_estimation_model.matching_algorithm import cross_view_fish_matching
from PCA.get_skeleton import extract_skeleton
from weight_estimation_model.get_3D_coordinate import StereoSpine3DReconstructor
from weight_estimation_model.weight_pretrained_model import estimate_fish_weight
from PCA.PCA_complete import batch_repair_fish_skeletons
from yolo_model.multi_task_yolo import FishPredictor
from data_handle.visualize import visualize_fishes_weight
#侧视图像输入专用YOLO
model = YOLO("")
side_results=model.predict("")
side_fishes = []
#每条鱼实例化
for i, (box, kpts) in enumerate(zip(side_results[0].boxes.xyxy, side_results[0].keypoints.xy)):
    x1, y1, x2, y2 = box.tolist()
    top_x, top_y = kpts[0].tolist()  # 第一个关键点：背部
    bottom_x, bottom_y = kpts[1].tolist()  # 第二个关键点：腹部

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
re3d = StereoSpine3DReconstructor()#输入参数
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