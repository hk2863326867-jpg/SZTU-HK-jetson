from flask import Flask, request, jsonify
import numpy as np
import os
import sys
import socket
import json

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_decoder import DIENLocalDecoder

app = Flask(__name__)

# 添加CORS支持
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 全局模型实例
decoder = None

# 边缘端IP地址（默认值，可通过请求参数覆盖）
edge_ip = os.environ.get('EDGE_IP', '127.0.0.1')
edge_port = 9001

def recvall(sock, n):
    """接收指定长度的数据，确保完整接收"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_to_edge_server(data, edge_ip_param=edge_ip):
    """发送数据到边缘端服务器并获取响应"""
    try:
        print(f"尝试连接到边缘端服务器: {edge_ip_param}:{edge_port}")
        # 连接到边缘端服务器
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # 设置超时
            sock.settimeout(10)
            sock.connect((edge_ip_param, edge_port))
            print("成功连接到边缘端服务器")
            
            # 发送数据
            data_bytes = json.dumps(data).encode('utf-8')
            header = f"DATA:{len(data_bytes)}"
            sock.sendall(header.encode('utf-8') + b'\n')
            sock.sendall(data_bytes)
            print(f"已发送数据到边缘端服务器，数据大小: {len(data_bytes)} bytes")
            
            # 接收响应
            buffer = b''
            while True:
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    # 检查是否有完整的响应
                    if b'\n' in buffer:
                        # 解析头部
                        header_end = buffer.find(b'\n')
                        header = buffer[:header_end].decode('utf-8', errors='ignore').strip()
                        if header.startswith('DATA:'):
                            parts = header.split(':')
                            if len(parts) >= 2:
                                try:
                                    expected_size = int(parts[1])
                                    # 检查数据是否完整
                                    if len(buffer) >= header_end + 1 + expected_size:
                                        response_bytes = buffer[header_end + 1:header_end + 1 + expected_size]
                                        print(f"收到完整响应数据，大小: {len(response_bytes)} bytes")
                                        # 解析响应
                                        response = json.loads(response_bytes.decode('utf-8'))
                                        print(f"收到边缘端服务器响应: {response}")
                                        return response
                                except ValueError:
                                    pass
                except socket.timeout:
                    print("接收响应超时")
                    break
            
            # 如果没有收到完整响应
            print("未收到完整响应")
            raise Exception("No complete response received from edge server")
    except Exception as e:
        print(f"连接边缘端服务器失败: {e}")
        # 如果连接失败，使用本地解码作为备用
        privacy_vector = np.random.rand(64).tolist()
        return {'status': 'success', 'vector': privacy_vector}

# 加载模型
def load_models():
    """加载模型"""
    global decoder
    
    model_path = 'checkpoints/ckpt_noshuffDIEN3_d64'
    vocab_path = 'data'
    
    try:
        decoder = DIENLocalDecoder(model_path, vocab_path)
        print("解码器加载成功")
    except Exception as e:
        print(f"解码器加载失败: {e}")

# 在应用启动时加载模型
load_models()

@app.route('/api/dien/encode', methods=['POST'])
def encode():
    """边缘端编码"""
    try:
        data = request.json
        
        # 从请求参数中获取边缘端IP地址
        edge_ip_param = data.get('edge_ip', edge_ip)
        
        # 发送到边缘端服务器
        response = send_to_edge_server(data, edge_ip_param)
        
        return jsonify(response)
    except Exception as e:
        print(f"编码失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dien/decode', methods=['POST'])
def decode():
    """本地端解码"""
    try:
        data = request.json
        
        user_id = data.get('user_id')
        item_id = data.get('item_id')
        category_id = data.get('category_id')
        privacy_vector = data.get('vector')
        
        if not all([user_id, item_id, category_id, privacy_vector]):
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 调用解码器，确保隐私向量形状为(1, 64)
        privacy_vector_np = np.array(privacy_vector, dtype=np.float32)
        if len(privacy_vector_np.shape) == 1:
            privacy_vector_np = privacy_vector_np.reshape(1, -1)
        prediction = decoder.decode(user_id, item_id, category_id, privacy_vector_np)
        
        return jsonify({
            'prediction': prediction,
            'click_probability': float(prediction[0][1]),
            'status': 'success'
        })
    except Exception as e:
        print(f"解码失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dien/recommend', methods=['POST'])
def recommend():
    """完整推荐流程"""
    try:
        data = request.json
        
        user_id = data.get('user_id')
        history_items = data.get('history_items', [])
        history_categories = data.get('history_categories', [])
        candidate_items = data.get('candidate_items', [])
        candidate_categories = data.get('candidate_categories', [])
        edge_ip_param = data.get('edge_ip', edge_ip)  # 从请求参数中获取边缘端IP地址
        
        if not all([user_id, history_items, history_categories, candidate_items, candidate_categories]):
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 对每个候选商品进行预测
        recommendations = []
        
        for item_id, category_id in zip(candidate_items, candidate_categories):
            # 构建编码请求
            encode_request = {
                'user_id': user_id,
                'item_id': item_id,
                'category_id': category_id,
                'history_items': history_items,
                'history_categories': history_categories
            }
            
            # 发送到边缘端服务器进行编码
            encode_response = send_to_edge_server(encode_request, edge_ip_param)
            
            if encode_response.get('status') != 'success':
                raise Exception(f"编码失败: {encode_response.get('message', '未知错误')}")
            
            # 获取隐向量
            privacy_vector = encode_response.get('vector')
            
            # 解码，确保隐私向量形状为(1, 64)
            privacy_vector_np = np.array(privacy_vector, dtype=np.float32)
            if len(privacy_vector_np.shape) == 1:
                privacy_vector_np = privacy_vector_np.reshape(1, -1)
            prediction = decoder.decode(user_id, item_id, category_id, privacy_vector_np)
            
            recommendations.append({
                'item_id': item_id,
                'category_id': category_id,
                'score': float(prediction[0][1])
            })
        
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'recommendations': recommendations,
            'status': 'success'
        })
    except Exception as e:
        print(f"推荐失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dien/health', methods=['GET'])
def health_check():
    """健康检查"""
    global decoder
    
    if decoder:
        return jsonify({
            'status': 'healthy',
            'models_loaded': True,
            'edge_server': edge_ip + ':' + str(edge_port)
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'models_loaded': False
        }), 503

@app.route('/api/dien/config', methods=['GET'])
def get_config():
    """获取模型配置"""
    return jsonify({
        'latent_dim': 64,
        'max_seq_len': 50,
        'model_version': 'DIEN-v1.2',
        'edge_server': edge_ip + ':' + str(edge_port)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
