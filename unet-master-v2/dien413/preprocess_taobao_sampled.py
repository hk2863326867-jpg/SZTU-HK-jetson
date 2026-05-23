import os
import pickle
import csv
from collections import defaultdict
import random

def preprocess_taobao_sampled(input_file, output_dir, max_sequence_length=50, sample_size=1000000):
    """
    预处理淘宝用户行为数据集（采样版本，避免占用太多空间）
    """
    print(f"Processing {input_file} with sampling (sample_size={sample_size})...")
    print("This will create a smaller dataset for training...")
    
    # 步骤1：第一次扫描，采样数据
    print("Step 1: Sampling data...")
    sampled_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i % 5000000 == 0 and i > 0:
                print(f"Scanned {i} lines, sampled {len(sampled_data)} so far...")
            if len(row) != 5:
                continue
            # 随机采样
            if len(sampled_data) < sample_size or random.random() < 0.01:
                sampled_data.append(row)
                if len(sampled_data) >= sample_size:
                    # 移除一些旧数据以保持采样数量
                    if random.random() < 0.1:
                        sampled_data.pop(random.randint(0, len(sampled_data)//2))
    
    print(f"Sampled {len(sampled_data)} lines from the dataset")
    
    # 步骤2：构建词汇表
    print("Step 2: Building vocabulary...")
    user_set = set()
    item_set = set()
    category_set = set()
    
    for row in sampled_data:
        user_id, item_id, category_id, behavior, timestamp = row
        user_set.add(user_id)
        item_set.add(item_id)
        category_set.add(category_id)
    
    # 生成词汇表
    user_voc = {user: idx+1 for idx, user in enumerate(user_set)}
    item_voc = {item: idx+1 for idx, item in enumerate(item_set)}
    category_voc = {cat: idx+1 for idx, cat in enumerate(category_set)}
    
    # 保存词汇表
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'uid_voc.pkl'), 'wb') as f:
        pickle.dump(user_voc, f)
    with open(os.path.join(output_dir, 'mid_voc.pkl'), 'wb') as f:
        pickle.dump(item_voc, f)
    with open(os.path.join(output_dir, 'cat_voc.pkl'), 'wb') as f:
        pickle.dump(category_voc, f)
    
    print(f"Vocabulary sizes:")
    print(f"  Users: {len(user_voc)}")
    print(f"  Items: {len(item_voc)}")
    print(f"  Categories: {len(category_voc)}")
    
    # 步骤3：构建用户序列并生成训练/测试数据
    print("Step 3: Building user sequences and generating data...")
    user_sequences = defaultdict(list)
    
    for row in sampled_data:
        user_id, item_id, category_id, behavior, timestamp = row
        user_sequences[user_id].append({
            'item_id': item_id,
            'category_id': category_id,
            'behavior': behavior,
            'timestamp': timestamp
        })
    
    # 打开输出文件
    train_file = open(os.path.join(output_dir, 'local_train_splitByUser'), 'w', encoding='utf-8')
    test_file = open(os.path.join(output_dir, 'local_test_splitByUser'), 'w', encoding='utf-8')
    
    train_count = 0
    test_count = 0
    
    for user_id, sequences in user_sequences.items():
        if len(sequences) < 2:
            continue
        
        # 按时间戳排序
        sequences.sort(key=lambda x: x['timestamp'])
        
        for i in range(1, len(sequences)):
            # 历史行为
            history = sequences[:i]
            # 当前行为
            current = sequences[i]
            
            # 构建序列
            history_items = [item['item_id'] for item in history]
            history_cats = [item['category_id'] for item in history]
            
            # 限制序列长度
            if len(history_items) > max_sequence_length:
                history_items = history_items[-max_sequence_length:]
                history_cats = history_cats[-max_sequence_length:]
            
            # 转换为ID
            user_id_enc = user_voc.get(user_id, 0)
            item_id_enc = item_voc.get(current['item_id'], 0)
            cat_id_enc = category_voc.get(current['category_id'], 0)
            history_items_enc = [item_voc.get(item, 0) for item in history_items]
            history_cats_enc = [category_voc.get(cat, 0) for cat in history_cats]
            
            # 标签：是否点击
            label = 1 if current['behavior'] == 'pv' else 0
            
            # 转换为字符串格式
            hist_items_str = ','.join(map(str, history_items_enc))
            hist_cats_str = ','.join(map(str, history_cats_enc))
            line = f"{user_id_enc}\t{item_id_enc}\t{cat_id_enc}\t{hist_items_str}\t{hist_cats_str}\t{label}\n"
            
            # 80%作为训练数据，20%作为测试数据
            if i < int(len(sequences) * 0.8):
                train_file.write(line)
                train_count += 1
            else:
                test_file.write(line)
                test_count += 1
    
    # 关闭文件
    train_file.close()
    test_file.close()
    
    print(f"Processing completed!")
    print(f"Train samples: {train_count}")
    print(f"Test samples: {test_count}")
    
    # 检查输出文件大小
    train_size = os.path.getsize(os.path.join(output_dir, 'local_train_splitByUser')) / (1024*1024)
    test_size = os.path.getsize(os.path.join(output_dir, 'local_test_splitByUser')) / (1024*1024)
    print(f"Train file size: {train_size:.2f} MB")
    print(f"Test file size: {test_size:.2f} MB")
    print(f"Total data size: {train_size + test_size:.2f} MB")

if __name__ == "__main__":
    random.seed(42)  # 设置随机种子以保证可重复性
    input_file = "d:\\OneDrive\\桌面\\github\\wangzeyu\\dien413\\UserBehavior.csv"
    output_dir = "d:\\OneDrive\\桌面\\github\\wangzeyu\\dien413\\data"
    preprocess_taobao_sampled(input_file, output_dir, sample_size=1000000)