import os
import json


def batch_clean_json_labels(json_folder, old_label="bicycle", new_label="bike"):
    print(f"🔬 启动工业级数据清洗程序：标签映射 [{old_label}] -> [{new_label}]")

    # 1. 找到所有 json 文件
    if not os.path.exists(json_folder):
        print(f"❌ 找不到文件夹：{json_folder}")
        return

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    print(f"📂 共发现 {len(json_files)} 个标注文件，准备清洗...")

    modified_count = 0
    box_count = 0

    # 2. 遍历并篡改底层数据
    for json_name in json_files:
        json_path = os.path.join(json_folder, json_name)

        # 读取 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        is_modified = False

        # 遍历里面的所有检测框(shapes)
        if 'shapes' in data:
            for shape in data['shapes']:
                if shape['label'] == old_label:
                    shape['label'] = new_label
                    is_modified = True
                    box_count += 1

        # 如果这张图被修改过，就覆盖保存回去
        if is_modified:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            modified_count += 1

    print("\n🎉 清洗彻底完工！")
    print(f"✅ 共修正了 {modified_count} 张图片中的 {box_count} 个错误标签。")
    print("🔄 现在你可以重新打开 X-AnyLabeling，你会发现它们全变成 bike 了！")


if __name__ == '__main__':
    # ================= 极简配置区 =================
    # 把你正在用 X-AnyLabeling 标注的那个图片文件夹路径填在这
    # （因为 X-AnyLabeling 默认把 json 和图片保存在同一个文件夹）
    TARGET_DIR = os.environ.get("LABEL_JSON_DIR", "data_to_clean")
    # ==========================================

    batch_clean_json_labels(TARGET_DIR)
