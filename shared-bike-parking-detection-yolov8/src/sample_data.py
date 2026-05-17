import os
import random
import shutil


def random_sample_dataset(source_dir, dest_dir, sample_size=150, seed=42):
    print("🔬 启动工业级数据采样程序...")

    # 固定随机种子，确保采样可复现
    random.seed(seed)

    # 1. 确保来源文件夹存在
    if not os.path.exists(source_dir):
        print(f"❌ 找不到来源文件夹：{source_dir}")
        return

    # 2. 找出所有合法的图片文件
    valid_extensions = ('.jpg', '.png', '.jpeg')
    all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)]

    total_images = len(all_images)
    print(f"📂 在来源文件夹中发现了 {total_images} 张图片。")

    if total_images == 0:
        print("❌ 来源文件夹里没有图片！")
        return

    # 3. 检查数量是否足够
    if total_images < sample_size:
        print(f"⚠️ 警告：总图片数 ({total_images}) 小于你想抽取的数量 ({sample_size})。将提取所有图片。")
        sample_size = total_images

    # 4. 核心：纯随机抽签算法
    print(f"🎲 正在随机抽取 {sample_size} 张高质量样本...")
    sampled_images = random.sample(all_images, sample_size)

    # 5. 创建输出文件夹并执行物理拷贝
    os.makedirs(dest_dir, exist_ok=True)

    print("⏳ 正在复制文件到清洗区...")
    for i, img_name in enumerate(sampled_images):
        src_path = os.path.join(source_dir, img_name)
        # 给图片改个名，打上“野生”标签，方便你区分
        dest_name = f"wild_bike_{i + 1:03d}.jpg"
        dest_path = os.path.join(dest_dir, dest_name)

        shutil.copy(src_path, dest_path)

    print(f"\n🎉 采样彻底完工！")
    print(f"✅ 成功抽取了 {sample_size} 张图片，并已重命名为 wild_bike_xxx.jpg")
    print(f"📁 请前往文件夹查看: {os.path.abspath(dest_dir)}")

    # 导出采样清单
    manifest_path = os.path.join(dest_dir, 'sample_manifest.txt')
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        mf.write(f"# 数据采样清单 (seed={seed}, sample_size={sample_size})\n")
        mf.write(f"# 来源: {source_dir}\n\n")
        for orig_name, dest_name in zip(sampled_images,
                                         [f'wild_bike_{i+1:03d}.jpg' for i in range(len(sampled_images))]):
            mf.write(f"{dest_name}\t{orig_name}\n")
    print(f"📋 采样清单已导出: {manifest_path}")


if __name__ == '__main__':
    # ================= 配置区 =================
    # 1. 把你那 3000 张开源单车图片，随便找个地方解压，把文件夹路径填在这里：
    # （如果是 Windows 路径，记得前面加个小写字母 r，防止斜杠报错）
    SOURCE_DATA = os.environ.get("SOURCE_DATA_DIR", "datasets/external/images")

    # 2. 抽出来的 150 张图，会自动放进你项目里的这个新文件夹
    CLEAN_DATA = os.environ.get("CLEAN_DATA_DIR", "data_to_clean")

    # 3. 严格控制的采样数量
    SAMPLE_AMOUNT = 700
    # ==========================================

    random_sample_dataset(SOURCE_DATA, CLEAN_DATA, SAMPLE_AMOUNT)
