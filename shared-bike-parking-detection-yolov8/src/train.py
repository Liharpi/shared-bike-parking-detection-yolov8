import os

from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练模型 (对于 6GB 显存，跑 nano 版本毫无压力，速度极快)
    model = YOLO('yolov8n.pt')

    print("🚀 拯救者引擎启动，准备切入深度学习状态...")
    results = model.train(
        data=os.environ.get("DATA_YAML", "datasets/YOLO_Master_Dataset/data.yaml"),
        epochs=100,
        imgsz=640,

        # === 针对 RTX 3060 6GB 显存的定制调优 ===
        batch=16,  # 6GB 显存的绝对安全线。如果设为 32 有小概率会在验证阶段爆显存中断。
        device=0,  # 强制指定使用 Nvidia 独显 (避开 CPU 自带的核心核显)

        # === 针对 i7-12700H (14核20线程) 的定制调优 ===
        workers=8,  # 你的 CPU 极其强大，开 8 个子线程去高速读取 SSD 里的图片，绝对喂得饱 3060

        # === 针对 16GB 内存的保护 ===
        cache=False,  # 坚决不使用 'ram' 缓存模式，否则 3000 张图很容易把你 16G 内存吃满导致系统卡死

        project='runs/detect',
        name='bike_parking_v1'
    )
    print("🎉 训练圆满结束！")
