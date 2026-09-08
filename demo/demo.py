import yaml
import argparse

from fish_weight_estimation.fish_camera_class import SideFishData
from ultralytics import YOLO
def parse_args():
    parser = argparse.ArgumentParser("双视角鱼群体重检测")
    parser.add_argument("--top_img", type=str, help="俯视图像")
    parser.add_argument("--side_fish_data", type=str, help="侧视图像")
    parser.add_argument("--model",default="best.pt",type=str,help="输入侧视YOLO模型,不输入则默认")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    side_results = model.predict(args.side_img)
    side_fishes = []
    # 每条鱼实例化
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
    print("已预测完成，结果已保存")









if __name__ == '__main__':
    main()