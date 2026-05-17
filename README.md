# Shared Bike Parking Detection System

## 项目简介

本项目面向共享单车乱停、倒地、越界和占用通道等场景，基于 YOLOv8 实现共享单车、倒地单车和停车区域检测，并结合规则判定完成违规识别、抓拍和历史记录管理。

## 技术栈

- Python
- YOLOv8
- PyTorch
- OpenCV
- PyQt5
- SQLite

## 核心功能

- 检测 bike、fallen_bike、parking_area 三类目标
- 支持图片、视频和摄像头输入
- 支持违规抓拍与历史记录查询
- 使用 AABB 交集比例判断车辆是否停入规范区域
- 使用 5 秒防抖机制降低误报
- 使用 SQLite 保存违规记录

## 项目结构

```text
shared-bike-parking-detection-yolov8/
├── src/
├── screenshots/
├── docs/
├── requirements.txt
├── README.md
└── .gitignore
