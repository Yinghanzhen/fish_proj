from ultralytics import YOLO
if __name__ == '__main__':

    model = YOLO(r"fish_weight_estimation/yolo_model/yolo26_att_best.yaml")
    model.train(
        data=r"fish_weight_estimation/yolo_model/fish.yaml",
        batch=4,
        epochs=200,
        imgsz=640,
        workers=4,
        device=0,
        amp=False,
        hsv_h=0.014,
        hsv_s = 0.645,
        hsv_v = 0.4,
        fliplr = 0.5,
        scale = 0.6,
        translate = 0.1,
        mosaic = 1.0,
        weight_decay=0.0005,
        dropout=0.1,
        optimizer='MuSGD',
        pretrained="yolo26n.pt",
    )