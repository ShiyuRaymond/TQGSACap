import json
import random
from collections import defaultdict

def filter_one_caption_per_video(input_json_path, output_json_path, seed=42):
    random.seed(seed)

    # 读取原始数据
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # 将同一 video_id 的 caption 聚集到一起
    video_caption_map = defaultdict(list)
    for item in data:
        video_id = item['video_id']
        caption = item['caption']
        video_caption_map[video_id].append(caption)

    # 为每个 video 随机保留一条 caption
    filtered_data = []
    for video_id, captions in video_caption_map.items():
        chosen_caption = random.choice(captions)
        filtered_data.append({"video_id": video_id, "caption": chosen_caption})

    # 写入新的 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"✅ 完成：原始样本数 {len(data)}，过滤后样本数 {len(filtered_data)}")


filter_one_caption_per_video(
    input_json_path="/workspace/YOLO-World/scripts/yoloworld_lightning/data/msvd/anno/train_cleaned.json",
    output_json_path="/workspace/YOLO-World/scripts/yoloworld_lightning/data/msvd/anno/train_filterd.json"
)
