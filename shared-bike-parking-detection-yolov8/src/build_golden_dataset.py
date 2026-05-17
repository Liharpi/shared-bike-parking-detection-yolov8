import os
import random
import shutil


def build_golden_dataset_v2(core_dir, extra_img_dir, extra_txt_dir, final_dir, val_ratio=0.15, seed=42):
    print("🔥 启动黄金验证集装配流水线 (V2.0 强悍版)...")

    # 固定随机种子，确保实验可复现
    random.seed(seed)

    # 1. 建立标准的 YOLO 远征军阵型
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        os.makedirs(os.path.join(final_dir, d), exist_ok=True)

    # ================= 阶段一：暴力解析混装的核心数据 =================
    print("\n📦 [阶段一] 正在拆解混装的核心实拍与 3D 数据...")

    # 因为图和标签混在一起，所以来源都在 core_dir
    all_files = os.listdir(core_dir)
    core_images = [f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg'))]

    valid_core = []
    for img in core_images:
        txt_name = os.path.splitext(img)[0] + '.txt'
        # 直接在同级目录下找对应的 txt
        if txt_name in all_files:
            valid_core.append((img, txt_name))

    print(f"🔍 成功从混装堆里捞出 {len(valid_core)} 对完美数据！")

    # 打乱并切分
    random.shuffle(valid_core)
    val_size = int(len(valid_core) * val_ratio)
    val_data = valid_core[:val_size]
    train_core_data = valid_core[val_size:]

    # 搬运核心数据
    def copy_data(data_list, subset):
        for img, txt in data_list:
            # 图和标签都从 core_dir 拿，但存入不同的子文件夹
            shutil.copy(os.path.join(core_dir, img), os.path.join(final_dir, f'images/{subset}', img))
            shutil.copy(os.path.join(core_dir, txt), os.path.join(final_dir, f'labels/{subset}', txt))

    copy_data(val_data, 'val')
    print(f"✅ 成功部署 [黄金验证集]: {len(val_data)} 张极品混合图！(已被物理隔离，严禁污染)")
    copy_data(train_core_data, 'train')
    print(f"✅ 成功部署 [核心训练底座]: {len(train_core_data)} 张！")

    # ================= 阶段二：注入网图兴奋剂 =================
    print("\n💉 [阶段二] 正在全量注入 2600+ 张单车网图到训练集...")
    extra_images = [f for f in os.listdir(extra_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    extra_count = 0
    for img in extra_images:
        txt_name = os.path.splitext(img)[0] + '.txt'
        if os.path.exists(os.path.join(extra_txt_dir, txt_name)):
            shutil.copy(os.path.join(extra_img_dir, img), os.path.join(final_dir, f'images/train', img))
            shutil.copy(os.path.join(extra_txt_dir, txt_name), os.path.join(final_dir, f'labels/train', txt_name))
            extra_count += 1

    print(f"✅ 成功注入 [单车背景库]: {extra_count} 张！")

    # 导出 train/val 文件清单，保证后续实验可复现
    manifest_path = os.path.join(final_dir, 'data_split_manifest.txt')
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        mf.write(f"# 数据集划分清单 (seed={seed}, val_ratio={val_ratio})\n")
        mf.write(f"# 生成时间: {__import__('datetime').datetime.now()}\n")
        mf.write(f"# val: {len(val_data)} 张, train_core: {len(train_core_data)} 张, train_extra: {extra_count} 张\n\n")
        for subset, data_list in [('val', val_data), ('train_core', train_core_data)]:
            for img, txt in data_list:
                mf.write(f"{subset}\t{img}\t{txt}\n")
        for img in extra_images:
            txt_name = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(os.path.join(extra_txt_dir, txt_name)):
                mf.write(f"train_extra\t{img}\t{txt_name}\n")
    print(f"📋 数据划分清单已导出: {manifest_path}")

    print(f"\n🎉 终极装配完成！你的大军已集结在: {os.path.abspath(final_dir)}")


if __name__ == '__main__':
    # ================= 终极配置区 (不用动，就用你刚才配好的) =================
    CORE_DIR = os.environ.get("CORE_DATASET_DIR", "datasets/Core_Dataset")
    EXTRA_IMG_DIR = os.environ.get("EXTRA_IMAGE_DIR", "datasets/external/images")
    EXTRA_TXT_DIR = os.environ.get("EXTRA_LABEL_DIR", "datasets/external/labels")
    FINAL_YOLO_DIR = os.environ.get("FINAL_YOLO_DIR", "datasets/YOLO_Master_Dataset")
    # ====================================================================
    build_golden_dataset_v2(CORE_DIR, EXTRA_IMG_DIR, EXTRA_TXT_DIR, FINAL_YOLO_DIR)
