import os
from PIL import Image


def batch_crop_watermark(source_folder, output_folder, crop_bottom_pixels=60):
    print(f"🔪 启动工业级数据清洗：批量裁边去水印...")

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片
    valid_extensions = ('.jpg', '.png', '.jpeg')
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]

    if not images:
        print("❌ 来源文件夹里没有找到图片！")
        return

    print(f"📂 共发现 {len(images)} 张图片，开始手起刀落...")

    processed_count = 0
    for img_name in images:
        img_path = os.path.join(source_folder, img_name)
        out_path = os.path.join(output_folder, img_name)

        try:
            # 打开图片
            with Image.open(img_path) as img:
                width, height = img.size

                # 定义裁剪区域：(左, 上, 右, 下)
                # 左上角保持 (0,0)，右边保持宽度，下边减去水印高度
                crop_area = (0, 0, width, height - crop_bottom_pixels)

                # 执行裁剪并保存
                cropped_img = img.crop(crop_area)
                cropped_img.save(out_path)
                processed_count += 1

        except Exception as e:
            print(f"⚠️ 处理图片 {img_name} 时出错: {e}")

    print("\n🎉 裁切彻底完工！")
    print(f"✅ 成功去除了 {processed_count} 张图片的水印，纯净版已生成！")
    print(f"📁 请前往查看: {os.path.abspath(output_folder)}")


if __name__ == '__main__':
    # ================= 配置区 =================
    # 1. 把你从网站上下载下来的带有水印的 50 张图片放在这个文件夹里
    INPUT_DIR = os.environ.get("WATERMARK_INPUT_DIR", "data_to_clean")

    # 2. 裁切掉水印后的干净图片会存放在这里
    OUTPUT_DIR = os.environ.get("WATERMARK_OUTPUT_DIR", "liblib_clean_images")

    # 3. 切掉底部多少像素？(LiblibAI的水印一般切掉底部 50-80 像素就干净了)
    # 如果运行完发现水印还在，就把这个数字改大点(比如 80)；如果切多了，就改小点。
    CROP_PIXELS = 80
    # ==========================================

    batch_crop_watermark(INPUT_DIR, OUTPUT_DIR, CROP_PIXELS)
