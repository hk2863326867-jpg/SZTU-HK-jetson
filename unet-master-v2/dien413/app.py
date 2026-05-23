from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# 加载词汇表
def load_vocab():
    data_dir = "d:\\OneDrive\\桌面\\github\\wangzeyu\\dien413\\data"
    with open(os.path.join(data_dir, 'uid_voc.pkl'), 'rb') as f:
        uid_voc = pickle.load(f)
    with open(os.path.join(data_dir, 'mid_voc.pkl'), 'rb') as f:
        mid_voc = pickle.load(f)
    with open(os.path.join(data_dir, 'cat_voc.pkl'), 'rb') as f:
        cat_voc = pickle.load(f)
    return uid_voc, mid_voc, cat_voc

# 加载模型
def load_model():
    model_path = "D:\\dien_checkpoints\\best\\ckpt_noshuffDIEN3_d64"
    # 使用tf.compat.v1
    tf.compat.v1.disable_v2_behavior()
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
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
    
    # 获取模型输入输出张量
    uid = graph.get_tensor_by_name('InputLayer/uid:0')
    iid = graph.get_tensor_by_name('InputLayer/iid:0')
    cat = graph.get_tensor_by_name('InputLayer/cat:0')
    hist_i = graph.get_tensor_by_name('InputLayer/hist_i:0')
    hist_c = graph.get_tensor_by_name('InputLayer/hist_c:0')
    y_hat = graph.get_tensor_by_name('FCN/y_hat:0')
    
    # 执行预测
    feed_dict = {
        uid: [user_id_enc],
        iid: [item_id_enc],
        cat: [category_id_enc],
        hist_i: [history_items_enc],
        hist_c: [history_cats_enc]
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

# 加载模型（全局变量，避免每次请求都加载）
uid_voc, mid_voc, cat_voc = load_vocab()
sess, graph = load_model()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    history_items = data.get('history_items', [])
    history_categories = data.get('history_categories', [])
    candidate_items = data.get('candidate_items', [])
    candidate_categories = data.get('candidate_categories', [])
    
    if not user_id or not candidate_items:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # 生成推荐
    predictions = batch_predict(
        sess, graph, user_id, candidate_items, candidate_categories,
        history_items, history_categories,
        uid_voc, mid_voc, cat_voc
    )
    
    # 构建响应
    recommendations = []
    for item_id, prob in predictions[:10]:  # 返回Top-10
        recommendations.append({
            'item_id': item_id,
            'score': float(prob)
        })
    
    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)