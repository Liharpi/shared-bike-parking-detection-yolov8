import os

from ultralytics import YOLO

if __name__ == '__main__':
    # 💥 关键点：不要用 yolov8n.pt，用你已经练好的最强大脑！
    model = YOLO(os.environ.get("MODEL_PATH", "weights/best.pt"))

    print("🚀 启动增量微调模式...")
    results = model.train(
        data=os.environ.get("DATA_YAML", "datasets/YOLO_Master_Dataset/data.yaml"),
        epochs=30,  # 因为它已经有基础了，再跑 30 轮左右就足够让它记住新框了
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project='runs/detect',
        name='bike_parking_finetune'
    )
