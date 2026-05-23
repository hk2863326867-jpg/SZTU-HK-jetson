import tensorflow as tf
import pickle
import os

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
    model_path = r"D:\dien_checkpoints\best\ckpt_noshuffDIEN3_d64"
    # 禁用TF2行为
    tf.compat.v1.disable_v2_behavior()
    # 创建会话
    sess = tf.compat.v1.Session()
    # 导入模型
    saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    # 获取图
    graph = tf.compat.v1.get_default_graph()
    return sess, graph

# 预测函数
def predict(sess, graph, user_id, item_id, category_id, history_items, history_cats, 
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
    else:
        # 填充0
        pad_length = maxlen - len(history_items_enc)
        history_items_enc = [0] * pad_length + history_items_enc
        history_cats_enc = [0] * pad_length + history_cats_enc
    
    # 获取模型输入占位符
    uid = graph.get_tensor_by_name('Inputs/uid_batch_ph:0')
    iid = graph.get_tensor_by_name('Inputs/mid_batch_ph:0')
    cat = graph.get_tensor_by_name('Inputs/cat_batch_ph:0')
    hist_i = graph.get_tensor_by_name('Inputs/mid_his_batch_ph:0')
    hist_c = graph.get_tensor_by_name('Inputs/cat_his_batch_ph:0')
    mask = graph.get_tensor_by_name('Inputs/mask:0')
    seq_len = graph.get_tensor_by_name('Inputs/seq_len_ph:0')
    # 负样本占位符（可以使用空值）
    noclk_mid = graph.get_tensor_by_name('Inputs/noclk_mid_batch_ph:0')
    noclk_cat = graph.get_tensor_by_name('Inputs/noclk_cat_batch_ph:0')
    
    # 获取输出张量
    # 从模型代码中，y_hat 是在 build_fcn_net 方法中定义的
    try:
        y_hat = graph.get_tensor_by_name('y_hat:0')
    except:
        # 打印所有操作，以便找到输出
        print("找不到 y_hat 张量，打印所有操作:")
        for op in graph.get_operations():
            if 'y_hat' in op.name:
                print(f"  {op.name}")
        raise
    
    # 计算序列长度和掩码
    seq_length = len([x for x in history_items_enc if x != 0])
    mask_values = [1 if x != 0 else 0 for x in history_items_enc]
    
    # 执行预测
    feed_dict = {
        uid: [user_id_enc],
        iid: [item_id_enc],
        cat: [category_id_enc],
        hist_i: [history_items_enc],
        hist_c: [history_cats_enc],
        mask: [mask_values],
        seq_len: [seq_length],
        # 负样本占位符（使用空值）
        noclk_mid: [[0] * maxlen],
        noclk_cat: [[0] * maxlen]
    }
    prediction = sess.run(y_hat, feed_dict=feed_dict)
    return prediction[0][1]  # 返回点击概率

# 批量预测
def batch_predict(sess, graph, user_id, item_ids, category_ids, history_items, history_cats,
                  uid_voc, mid_voc, cat_voc):
    predictions = []
    for item_id, category_id in zip(item_ids, category_ids):
        prob = predict(sess, graph, user_id, item_id, category_id, history_items, history_cats,
                      uid_voc, mid_voc, cat_voc)
        predictions.append((item_id, prob))
    # 按概率排序
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

if __name__ == "__main__":
    # 加载词汇表和模型
    print("加载词汇表...")
    uid_voc, mid_voc, cat_voc = load_vocab()
    print("加载模型...")
    sess, graph = load_model()
    
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
    predictions = batch_predict(sess, graph, user_id, candidate_items, candidate_categories,
                               user_history['items'], user_history['categories'],
                               uid_voc, mid_voc, cat_voc)
    
    # 输出结果
    print("\n推荐结果（按点击概率排序）:")
    for item_id, prob in predictions:
        print(f"商品ID: {item_id}, 预测点击概率: {prob:.4f}")
    
    # 关闭会话
    sess.close()