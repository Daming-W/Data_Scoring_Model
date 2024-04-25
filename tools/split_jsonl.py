import jsonlines
from tqdm import tqdm

def split_jsonl_with_progress(original_file, lines_per_file=100000):
    current_lines = []
    file_count = 1

    with jsonlines.open(original_file) as reader:
        # total_lines = sum(1 for _ in reader)  # 首先统计总行数
        # reader.seek(0)  # 重置文件指针到开始位置

        with tqdm(total=8800000, desc="Processing lines") as progress:
            for line in reader:
                current_lines.append(line)
                progress.update(1)  # 每处理一行就更新进度条
                if len(current_lines) == lines_per_file:
                    # 写入新文件
                    with jsonlines.open(f'part_{file_count}.jsonl', mode='w') as writer:
                        writer.write_all(current_lines)
                    current_lines = []
                    file_count += 1

            # 检查是否有剩余的行未写入
            if current_lines:
                with jsonlines.open(f'/mnt/share_disk/wangdaming/scoremodel_data/880w_jsonl_chunks/part_{file_count}.jsonl', mode='w') as writer:
                    writer.write_all(current_lines)

# 使用示例
split_jsonl_with_progress('/mnt/share_disk/LIV/datacomp/processed_data/880w_1088w_dedup_processed/v0.2/clip_large_stats.jsonl')
