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


if __name__ == "__main__":
    main()









if __name__ == '__main__':
    main()