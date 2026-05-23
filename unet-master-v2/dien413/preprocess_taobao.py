import os
import pickle
import csv
from collections import defaultdict

def preprocess_taobao(input_file, output_dir, max_sequence_length=50, batch_size=1000000):
    """
    预处理淘宝用户行为数据集（分批处理）
    """
    print(f"Processing {input_file}...")
    
    # 步骤1：第一次扫描，构建词汇表
    print("Step 1: Building vocabulary...")
    user_set = set()
    item_set = set()
    category_set = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i % 1000000 == 0 and i > 0:
                print(f"Processed {i} lines...")
            if len(row) != 5:
                continue
            user_id, item_id, category_id, behavior, timestamp = row
            user_set.add(user_id)
            item_set.add(item_id)
            category_set.add(category_id)
    
    # 生成词汇表
    user_voc = {user: idx+1 for idx, user in enumerate(user_set)}  # 0留作padding
    item_voc = {item: idx+1 for idx, item in enumerate(item_set)}  # 0留作padding
    category_voc = {cat: idx+1 for idx, cat in enumerate(category_set)}  # 0留作padding
    
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
    
    # 步骤2：第二次扫描，构建用户序列并生成训练/测试数据
    print("Step 2: Building user sequences and generating data...")
    user_sequences = defaultdict(list)
    
    # 打开输出文件
    train_file = open(os.path.join(output_dir, 'local_train_splitByUser'), 'w', encoding='utf-8')
    test_file = open(os.path.join(output_dir, 'local_test_splitByUser'), 'w', encoding='utf-8')
    
    def process_user_sequence(user_id, sequences):
        """处理单个用户的序列"""
        if len(sequences) < 2:
            return  # 至少需要2个行为才能生成样本
        
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
            
            # 标签：是否点击（这里简化处理，实际应该根据behavior字段）
            label = 1 if current['behavior'] == 'pv' else 0
            
            # 转换为字符串格式
            hist_items_str = ','.join(map(str, history_items_enc))
            hist_cats_str = ','.join(map(str, history_cats_enc))
            line = f"{user_id_enc}\t{item_id_enc}\t{cat_id_enc}\t{hist_items_str}\t{hist_cats_str}\t{label}\n"
            
            # 80%作为训练数据，20%作为测试数据
            if i < int(len(sequences) * 0.8):
                train_file.write(line)
            else:
                test_file.write(line)
    
    # 逐行读取数据，按用户ID分组
    current_user = None
    batch_count = 0
    processed_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 5:
                continue
            user_id, item_id, category_id, behavior, timestamp = row
            processed_lines += 1
            
            if processed_lines % 1000000 == 0:
                print(f"Processed {processed_lines} lines...")
            
            # 检查用户是否变化
            if user_id != current_user:
                # 处理之前的用户
                if current_user and current_user in user_sequences:
                    process_user_sequence(current_user, user_sequences[current_user])
                    del user_sequences[current_user]
                
                # 开始新用户
                current_user = user_id
            
            # 添加到用户序列
            user_sequences[user_id].append({
                'item_id': item_id,
                'category_id': category_id,
                'behavior': behavior,
                'timestamp': timestamp
            })
            
            # 定期清理内存
            if len(user_sequences) > 10000:
                # 处理并清理老用户
                users_to_remove = list(user_sequences.keys())[:5000]
                for user in users_to_remove:
                    if user != current_user:
                        process_user_sequence(user, user_sequences[user])
                        del user_sequences[user]
    
    # 处理最后一个用户
    if current_user and current_user in user_sequences:
        process_user_sequence(current_user, user_sequences[current_user])
    
    # 关闭文件
    train_file.close()
    test_file.close()
    
    # 统计数据量
    train_count = sum(1 for _ in open(os.path.join(output_dir, 'local_train_splitByUser'), 'r', encoding='utf-8'))
    test_count = sum(1 for _ in open(os.path.join(output_dir, 'local_test_splitByUser'), 'r', encoding='utf-8'))
    
    print(f"Processing completed!")
    print(f"Train samples: {train_count}")
    print(f"Test samples: {test_count}")

if __name__ == "__main__":
    input_file = "d:\\OneDrive\\桌面\\github\\wangzeyu\\dien413\\UserBehavior.csv"
    output_dir = "d:\\OneDrive\\桌面\\github\\wangzeyu\\dien413\\data"
    preprocess_taobao(input_file, output_dir)