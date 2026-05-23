import tensorflow as tf
import pickle
import os
import sys

# 添加 dien/script 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dien', 'script'))

from model import Model_DIN_V2_Gru_Vec_attGru_Neg

# 加载词汇表
def load_vocab():
    data_dir = r"d:\OneDrive\桌面\github\wangzeyu\dien413\data"
    with open(os.path.join(data_dir, 'uid_voc.pkl'), 'rb') as f:
        uid_voc = pickle.load(f)
    with open(os.path.join(data_dir, 'mid_voc.pkl'), 'rb') as f:
        mid_voc = pickle.load(f)
    with open(os.path.join(data_dir, 'cat_voc.pkl'), 'rb') as f:
        cat_voc = pickle.load(f)
    return uid_voc, mid_voc, cat_voc

# 加载模型
def load_model():
    # 禁用TF2行为
    tf.compat.v1.disable_v2_behavior()
    # 创建会话
    sess = tf.compat.v1.Session()
    
    # 加载词汇表，获取词汇表大小
    uid_voc, mid_voc, cat_voc = load_vocab()
    n_uid = len(uid_voc) + 1  # +1 for padding
    n_mid = len(mid_voc) + 1
    n_cat = len(cat_voc) + 1
    
    # 创建模型实例
    model = Model_DIN_V2_Gru_Vec_attGru_Neg(
        n_uid, n_mid, n_cat, 
        EMBEDDING_DIM=18, 
        HIDDEN_SIZE=12, 
        ATTENTION_SIZE=12,
        use_negsampling=True,
        latent_dim=64
    )
    
    # 初始化变量
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # 加载模型权重
    model_path = r"D:\dien_checkpoints\best\ckpt_noshuffDIEN3_d64"
    # 检查模型文件是否存在
    if os.path.exists(model_path + '.meta'):
        model.restore(sess, model_path)
        print(f"模型成功加载: {model_path}")
    else:
        print(f"警告: 模型文件不存在: {model_path}")
        print("使用随机初始化的模型参数")
    
    return sess, model, uid_voc, mid_voc, cat_voc

# 预测函数
def predict(sess, model, user_id, item_id, category_id, history_items, history_cats, 
            uid_voc, mid_voc, cat_voc):
    # 转换为编码
    user_id_enc = uid_voc.get(user_id, 0)
    item_id_enc = mid_voc.get(item_id, 0)
    category_id_enc = cat_voc.get(category_id, 0)
    history_items_enc = [mid_voc.get(item, 0) for item in history_items]
    history_cats_enc = [cat_voc.get(cat, 0) for cat in history_cats]
    
    # 填充或截断序列
    maxlen = 50
    if len(history_items_enc) > maxlen:
        history_items_enc = history_items_enc[-maxlen:]
        history_cats_enc = history_cats_enc[-maxlen:]
    
    # 计算序列长度和掩码
    seq_length = len(history_items_enc)
    mask_values = [1.0] * seq_length + [0.0] * (maxlen - seq_length)
    
    # 填充序列
    history_items_enc += [0] * (maxlen - seq_length)
    history_cats_enc += [0] * (maxlen - seq_length)
    
    # 负样本（使用空值）
    noclk_mid = [[0] * 3 for _ in range(maxlen)]  # 3 negative samples
    noclk_cat = [[0] * 3 for _ in range(maxlen)]
    
    # 目标值（用于计算损失，预测时可以随便设置）
    target = [[1.0, 0.0]]  # 假设点击
    
    # 准备输入数据
    inps = [
        [user_id_enc],  # uid
        [item_id_enc],  # mid
        [category_id_enc],  # cat
        [history_items_enc],  # mid_his
        [history_cats_enc],  # cat_his
        [mask_values],  # mask
        target,  # target
        [seq_length],  # seq_len
        [noclk_mid],  # noclk_mid
        [noclk_cat]   # noclk_cat
    ]
    
    # 执行预测
    probs, loss, accuracy, aux_loss = model.calculate(sess, inps)
    return probs[0][1]  # 返回点击概率

# 批量预测
def batch_predict(sess, model, user_id, item_ids, category_ids, history_items, history_cats,
                  uid_voc, mid_voc, cat_voc):
    predictions = []
    for item_id, category_id in zip(item_ids, category_ids):
        prob = predict(sess, model, user_id, item_id, category_id, history_items, history_cats,
                      uid_voc, mid_voc, cat_voc)
        predictions.append((item_id, prob))
    # 按概率排序
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

if __name__ == "__main__":
    # 加载模型和词汇表
    print("加载模型和词汇表...")
    sess, model, uid_voc, mid_voc, cat_voc = load_model()
    
    # 示例用户数据
    user_id = "1000001"  # 示例用户ID
    user_history = {
        'items': ['3108855', '2640191', '2203057'],  # 用户历史浏览商品
        'categories': ['794437', '794437', '794437']  # 对应商品类别
    }
    
    # 候选商品列表
    candidate_items = ['3108855', '2640191', '2203057', '2735466', '1522137']
    candidate_categories = ['794437', '794437', '794437', '1464116', '1464116']
    
    # 预测并排序
    print("生成推荐...")
    predictions = batch_predict(sess, model, user_id, candidate_items, candidate_categories,
                               user_history['items'], user_history['categories'],
                               uid_voc, mid_voc, cat_voc)
    
    # 输出结果
    print("\n推荐结果（按点击概率排序）:")
    for item_id, prob in predictions:
        print(f"商品ID: {item_id}, 预测点击概率: {prob:.4f}")
    
    # 关闭会话
    sess.close()