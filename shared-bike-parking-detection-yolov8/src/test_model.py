import os

import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载你的最强大脑 (请确保路径正确！)
    model = YOLO(os.environ.get("MODEL_PATH", "weights/best.pt"))

    # 2. 测试图片路径
    test_source = os.environ.get("TEST_SOURCE", "samples/example.jpg")

    print("👁️ 模型启动，正在扫描实景并绘制判定界面...")
    results = model.predict(source=test_source, conf=0.4)
    result = results[0]

    # 获取原始图片的 numpy 数组，准备在上面画画
    img = result.orig_img.copy()

    parking_areas = []
    bikes = []
    fallen_bikes = []

    # ================= 第一遍扫描：收集所有坐标 =================
    for box in result.boxes.data:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        cls_id = int(cls_id)

        if cls_id == 2:  # 停车区
            parking_areas.append([int(x1), int(y1), int(x2), int(y2)])
        elif cls_id == 0:  # 正常单车
            bikes.append([int(x1), int(y1), int(x2), int(y2)])
        elif cls_id == 1:  # 倒地单车
            fallen_bikes.append([int(x1), int(y1), int(x2), int(y2)])

    # ================= 第二遍扫描：开始在图片上画画 =================

    # 1. 画出所有的停车区 (蓝色粗框)
    for area in parking_areas:
        cv2.rectangle(img, (area[0], area[1]), (area[2], area[3]), (255, 0, 0), 3)
        cv2.putText(img, "Parking Area", (area[0], area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 2. 画出倒地单车 (紫色框，严重警告)
    for fallen in fallen_bikes:
        cv2.rectangle(img, (fallen[0], fallen[1]), (fallen[2], fallen[3]), (255, 0, 255), 3)
        cv2.putText(img, "WARNING: Fallen!", (fallen[0], fallen[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255),
                    2)

     # 3. 核心逻辑升级版：基于交集（Overlap）的容错判定
    for bike in bikes:
        bike_x1, bike_y1, bike_x2, bike_y2 = bike

        # 依然计算中心点（只是为了画图好看）
        cx = (bike_x1 + bike_x2) // 2
        cy = (bike_y1 + bike_y2) // 2

        is_safe = False
        for area in parking_areas:
            area_x1, area_y1, area_x2, area_y2 = area

            # 【算法升级】：计算两个框的重叠区域（Intersection）
            # 找到重叠矩形的左上角和右下角坐标
            inter_x1 = max(bike_x1, area_x1)
            inter_y1 = max(bike_y1, area_y1)
            inter_x2 = min(bike_x2, area_x2)
            inter_y2 = min(bike_y2, area_y2)

            # 如果重叠矩形的右下角坐标 > 左上角坐标，说明它们相交了！
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                # 只要碰到框，就认为是规范停车
                is_safe = True
                break

        if is_safe:
            # 安全：画绿色框
            cv2.rectangle(img, (bike_x1, bike_y1), (bike_x2, bike_y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(img, "Safe", (bike_x1, bike_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 违停：画红色框
            cv2.rectangle(img, (bike_x1, bike_y1), (bike_x2, bike_y2), (0, 0, 255), 3)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img, "ILLEGAL!", (bike_x1, bike_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ================= 终极展示 =================
    # 缩小图片以便在屏幕上显示 (如果原图是4K的话会显示不下)
    img_show = cv2.resize(img, (1280, 720))

    cv2.imshow("Smart Bike Parking Monitor V1.0", img_show)
    print("✅ 画面已生成！按键盘上的任意键关闭窗口。")
    cv2.waitKey(0)  # 暂停程序，等待用户按键
    cv2.destroyAllWindows()
