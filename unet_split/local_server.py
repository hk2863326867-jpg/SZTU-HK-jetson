# -*- coding: utf-8 -*-
import socket
import os
import json
from flask import Flask, request, jsonify
import base64
import numpy as np
import torch
from model_decoder import UNetDecoder
import skimage.io as io
import skimage.transform as trans

app = Flask(__name__)

def recvall(sock, n):
    """接收指定长度的数据，确保完整接收"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# 加载解码器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = UNetDecoder(out_channels=1).to(device)
decoder.load_state_dict(torch.load('saved_weights/decoder_weights.pth', map_location=device))
decoder.eval()
print(f"Decoder loaded on {device}")

# 发送图片到 Jetson 并接收特征
def send_image_to_jetson(image_path, jetson_ip):
    print(f"Sending image to Jetson at {jetson_ip}...")
    
    try:
        # 连接到 Jetson
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((jetson_ip, 9000))
            
            # 读取图片
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # 发送图片头信息
            filename = os.path.basename(image_path)
            header = f"IMAGE:{filename}:{len(image_data)}"
            sock.sendall(header.encode('utf-8') + b'\n')
            
            # 发送图片数据
            sock.sendall(image_data)
            sock.sendall(b'END_OF_IMAGE')
            
            # 接收特征
            print("Receiving features from Jetson...")
            
            # 接收特征数量
            num_features_bytes = recvall(sock, 4)
            if num_features_bytes is None:
                raise Exception("Failed to receive num_features")
            num_features = int.from_bytes(num_features_bytes, byteorder='little')
            print(f"Expected features: {num_features}")
            
            # 接收瓶颈层特征
            b0_bytes = recvall(sock, 4)
            b1_bytes = recvall(sock, 4)
            b2_bytes = recvall(sock, 4)
            b3_bytes = recvall(sock, 4)
            if any(x is None for x in [b0_bytes, b1_bytes, b2_bytes, b3_bytes]):
                raise Exception("Failed to receive bottleneck shape")
            b0 = int.from_bytes(b0_bytes, byteorder='little')
            b1 = int.from_bytes(b1_bytes, byteorder='little')
            b2 = int.from_bytes(b2_bytes, byteorder='little')
            b3 = int.from_bytes(b3_bytes, byteorder='little')
            bottleneck_shape = (b0, b1, b2, b3)
            
            bottleneck_size = b0 * b1 * b2 * b3 * 4  # float32
            bottleneck_bytes = recvall(sock, bottleneck_size)
            if bottleneck_bytes is None:
                raise Exception("Failed to receive bottleneck data")
            bottleneck_np = np.frombuffer(bottleneck_bytes, dtype=np.float32).reshape(bottleneck_shape)
            bottleneck = torch.from_numpy(bottleneck_np)
            print(f"Received bottleneck: {bottleneck_shape}")
            
            # 接收跳跃连接特征
            features = []
            for i in range(num_features):
                f0_bytes = recvall(sock, 4)
                f1_bytes = recvall(sock, 4)
                f2_bytes = recvall(sock, 4)
                f3_bytes = recvall(sock, 4)
                if any(x is None for x in [f0_bytes, f1_bytes, f2_bytes, f3_bytes]):
                    raise Exception(f"Failed to receive feature {i} shape")
                f0 = int.from_bytes(f0_bytes, byteorder='little')
                f1 = int.from_bytes(f1_bytes, byteorder='little')
                f2 = int.from_bytes(f2_bytes, byteorder='little')
                f3 = int.from_bytes(f3_bytes, byteorder='little')
                feature_shape = (f0, f1, f2, f3)
                
                feature_size = f0 * f1 * f2 * f3 * 4  # float32
                feature_bytes = recvall(sock, feature_size)
                if feature_bytes is None:
                    raise Exception(f"Failed to receive feature {i} data")
                feature_np = np.frombuffer(feature_bytes, dtype=np.float32).reshape(feature_shape)
                feature = torch.from_numpy(feature_np)
                features.append(feature)
                print(f"Received feature {i+1}: {feature_shape}")
            
            print("All features received successfully!")
            return bottleneck, features
            
    except Exception as e:
        print(f"Error communicating with Jetson: {e}")
        return None, None

# 后处理
def postprocess(output):
    output = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # output = 1 - output  # 反转输出（人物=亮色）
    output = (output * 255).astype(np.uint8)
    return output

# 保存结果
def save_result(mask, output_path):
    io.imsave(output_path, mask)
    print(f"Result saved to: {output_path}")

# 主端点
@app.route('/api/upload-to-jetson', methods=['POST'])
def upload_to_jetson():
    try:
        # 获取图片和 Jetson IP
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"})
        
        if 'jetsonIp' not in request.form:
            return jsonify({"success": False, "error": "No Jetson IP provided"})
        
        image = request.files['image']
        jetson_ip = request.form['jetsonIp']
        
        # 保存临时图片
        temp_path = f"temp_{image.filename}"
        image.save(temp_path)
        print(f"Received image: {image.filename}")
        
        # 发送到 Jetson 并获取特征
        bottleneck, features = send_image_to_jetson(temp_path, jetson_ip)
        
        if bottleneck is None or features is None:
            return jsonify({"success": False, "error": "Failed to get features from Jetson"})
        
        # 解码
        print("Running decoder...")
        bottleneck = bottleneck.to(device)
        features = [f.to(device) for f in features]
        
        with torch.no_grad():
            output = decoder(bottleneck, features)
        
        # 获取原始预测
        raw_mask = postprocess(output)
        
        # 获取阈值处理后的预测
        threshold_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        threshold = 0.7
        threshold_mask = (threshold_mask > threshold).astype(np.float32)
        
        # 将掩码调整回原始图片尺寸
        from PIL import Image
        with Image.open(temp_path) as original_img:
            original_size = original_img.size  # (width, height)
        
        # 调整尺寸
        raw_mask_resized = trans.resize(raw_mask, (original_size[1], original_size[0]))
        raw_mask_resized = (raw_mask_resized * 255).astype(np.uint8)
        
        threshold_mask_resized = trans.resize(threshold_mask, (original_size[1], original_size[0]))
        threshold_mask_resized = (threshold_mask_resized * 255).astype(np.uint8)
        
        # 保存结果
        raw_output_path = f"raw_output_{image.filename}"
        threshold_output_path = f"threshold_output_{image.filename}"
        
        save_result(raw_mask_resized, raw_output_path)
        save_result(threshold_mask_resized, threshold_output_path)
        
        # 转换为 base64
        with open(raw_output_path, 'rb') as f:
            raw_mask_data = f.read()
        raw_mask_base64 = base64.b64encode(raw_mask_data).decode('utf-8')
        
        with open(threshold_output_path, 'rb') as f:
            threshold_mask_data = f.read()
        threshold_mask_base64 = base64.b64encode(threshold_mask_data).decode('utf-8')
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)
        if os.path.exists(threshold_output_path):
            os.remove(threshold_output_path)
        
        return jsonify({
            "success": True,
            "raw_mask": raw_mask_base64,
            "threshold_mask": threshold_mask_base64
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)})

# 健康检查
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 启动服务器
    print("Starting local server...")
    print("Listening on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
