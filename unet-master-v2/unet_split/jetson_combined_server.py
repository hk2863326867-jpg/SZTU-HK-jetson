#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一Jetson服务器 - 整合WiFi设置和图像编码功能

功能：
1. WiFi参数设置服务（端口5000）
2. 图像编码服务（Socket端口9000）
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import socket
import sys
import os
import threading
import time
from datetime import datetime
import torch
import numpy as np
import skimage.io as io
import skimage.transform as trans
import re

# 导入WiFi控制器
from wifi_controller import WiFiController

# 创建WiFi控制器实例
wifi_controller = WiFiController()

# 图像编码配置
SAVE_DIR = '/home/nvidia/Pictures/images'
os.makedirs(SAVE_DIR, exist_ok=True)

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 加载编码器（使用 CPU 模式避免 CUDA 兼容性问题）
try:
    from model_encoder import UNetEncoder
    device = torch.device('cpu')
    encoder = UNetEncoder(in_channels=3).to(device)
    encoder.load_state_dict(torch.load('saved_weights/encoder_weights.pth', map_location=device))
    encoder.eval()
    print(f"Encoder loaded on {device}")
    ENCODER_LOADED = True
except Exception as e:
    print(f"Failed to load encoder: {e}")
    ENCODER_LOADED = False

# WiFi功能已移至wifi_controller.py

# ---------------------- 图像编码功能模块 ----------------------
def preprocess_image(image_path):
    print(f"Processing image: {image_path}")
    image = io.imread(image_path)
    
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    image = trans.resize(image, (256, 256), preserve_range=True)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image = torch.FloatTensor(image).unsqueeze(0)
    
    return image

def encode_image(image_path):
    print("Encoding image...")
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        bottleneck, features = encoder(image_tensor)
    
    print(f"Encoding completed. Bottleneck shape: {bottleneck.shape}")
    return bottleneck, features

def send_features(conn, bottleneck, features):
    print("Sending features...")
    
    # 转换为 NumPy 数组
    bottleneck_np = bottleneck.cpu().numpy()
    features_np = [f.cpu().numpy() for f in features]
    
    # 发送特征数量
    num_features = len(features_np)
    conn.sendall(num_features.to_bytes(4, byteorder='little'))
    
    # 发送瓶颈层特征
    bottleneck_shape = bottleneck_np.shape
    conn.sendall(bottleneck_shape[0].to_bytes(4, byteorder='little'))
    conn.sendall(bottleneck_shape[1].to_bytes(4, byteorder='little'))
    conn.sendall(bottleneck_shape[2].to_bytes(4, byteorder='little'))
    conn.sendall(bottleneck_shape[3].to_bytes(4, byteorder='little'))
    conn.sendall(bottleneck_np.tobytes())
    
    # 发送跳跃连接特征
    for i, feature in enumerate(features_np):
        feature_shape = feature.shape
        conn.sendall(feature_shape[0].to_bytes(4, byteorder='little'))
        conn.sendall(feature_shape[1].to_bytes(4, byteorder='little'))
        conn.sendall(feature_shape[2].to_bytes(4, byteorder='little'))
        conn.sendall(feature_shape[3].to_bytes(4, byteorder='little'))
        conn.sendall(feature.tobytes())
        print(f"Sent feature {i+1}: {feature_shape}")
    
    print("All features sent successfully!")

def handle_socket_client(conn, addr):
    """Handle socket client connection, receive image"""
    print(f"\n[+] New socket connection from {addr}")
    
    try:
        # Set timeout
        conn.settimeout(60)  # 增加超时时间
        
        # Receive data buffer
        buffer = b''
        expected_size = 0
        filename = None
        header_received = False
        
        while True:
            try:
                # Receive data
                chunk = conn.recv(4096)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Parse header info
                if not header_received and b'\n' in buffer:
                    # Find first line (header)
                    header_end = buffer.find(b'\n')
                    header = buffer[:header_end].decode('utf-8', errors='ignore').strip()
                    buffer = buffer[header_end + 1:]
                    
                    print(f"[*] Received header: {header}")
                    
                    # Parse header format: IMAGE:filename:size
                    if header.startswith('IMAGE:'):
                        parts = header.split(':')
                        if len(parts) >= 3:
                            filename = parts[1]
                            try:
                                expected_size = int(parts[2])
                                header_received = True
                                print(f"[*] Expecting image: {filename} ({expected_size} bytes)")
                            except ValueError:
                                print(f"[!] Invalid size in header: {parts[2]}")
                                break
                        else:
                            print(f"[!] Invalid header format")
                            break
                    else:
                        print(f"[!] Unknown header: {header}")
                        break
                
                # Check for end marker
                if header_received and b'END_OF_IMAGE' in buffer:
                    # Find end marker position
                    end_marker_pos = buffer.find(b'END_OF_IMAGE')
                    image_data = buffer[:end_marker_pos]
                    
                    # Save image
                    if filename and len(image_data) > 0:
                        # Generate filename with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_filename = f"{timestamp}_{filename}"
                        filepath = os.path.join(SAVE_DIR, safe_filename)
                        
                        # Save file
                        with open(filepath, 'wb') as f:
                            f.write(image_data)
                        
                        print(f"[+] Image saved: {filepath}")
                        print(f"[+] Size: {len(image_data)} bytes")
                        
                        # 编码图像
                        try:
                            bottleneck, features = encode_image(filepath)
                            
                            # 发送特征
                            send_features(conn, bottleneck, features)
                            
                            # 发送确认
                            conn.send(f"OK:Image processed and features sent\n".encode('utf-8'))
                        except Exception as e:
                            print(f"[!] Error processing image: {e}")
                            conn.send(f"ERROR:Failed to process image: {e}\n".encode('utf-8'))
                    
                    # Clear buffer for next image
                    buffer = b''
                    header_received = False
                    filename = None
                    expected_size = 0
                    
                # Prevent buffer overflow
                if len(buffer) > 20 * 1024 * 1024:  # 20MB limit
                    print("[!] Buffer too large, clearing...")
                    buffer = b''
                    header_received = False
                    
            except socket.timeout:
                print("[!] Receive timeout")
                break
            except Exception as e:
                print(f"[!] Error receiving data: {e}")
                break
                
    except Exception as e:
        print(f"[!] Client handler error: {e}")
    finally:
        conn.close()
        print(f"[-] Connection closed: {addr}")

# ---------------------- API路由 ----------------------
@app.route('/get_real_power')
def api_get_real_power():
    return jsonify({"power": wifi_controller.get_real_tx_power()})

@app.route('/get_real_channel')
def api_get_real_channel():
    return jsonify({"channel": wifi_controller.get_real_channel()})

@app.route('/get_connected_devices')
def api_get_connected_devices():
    return wifi_controller.get_connected_devices()

@app.route('/set_channel', methods=['POST'])
def api_set_channel():
    new_channel = request.form.get("channel", wifi_controller.get_real_channel())
    threading.Thread(target=wifi_controller.set_channel_async, args=(new_channel,)).start()
    return jsonify({"success": True, "message": "信道修改提交成功！"})

@app.route('/set_power', methods=['POST'])
def api_set_power():
    new_tx_power = request.form.get("tx_power", wifi_controller.get_real_tx_power())
    threading.Thread(target=wifi_controller.set_power_async, args=(new_tx_power,)).start()
    return jsonify({"success": True, "message": "功率修改提交成功！"})

# ---------------------- Socket服务器 ----------------------
def start_socket_server():
    HOST = ''
    PORT = 9000
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"[*] Socket server started on port {PORT}")
        
        while True:
            conn, addr = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_socket_client,
                args=(conn, addr),
                daemon=True
            )
            client_thread.start()
    except Exception as e:
        print(f"Socket server error: {e}")

# 启动socket服务器线程
socket_thread = threading.Thread(target=start_socket_server, daemon=True)
socket_thread.start()

# ---------------------- 页面路由（保留兼容） ----------------------
@app.route('/')
def index():
    wifi_params = {
        "ssid": "Board_Hotspot",
        "interface": "wlP1p1s0",
        "channel": wifi_controller.get_real_channel(),
        "tx_power_actual": wifi_controller.get_real_tx_power(),
        "mode": "AP"
    }
    connected_devices = api_get_connected_devices()
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>WiFi功率/信道配置面板</title>
        <style>
            body {{font-family: Arial, sans-serif; margin: 15px; line-height: 1.6; font-size: 14px;}}
            .card {{border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 15px; background: #f9f9f9;}}
            .card h2 {{margin-top: 0; font-size: 16px; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 6px;}}
            .config-group {{margin: 10px 0; padding: 8px; background: #e8f4f8; border-radius: 4px;}}
            label {{display: inline-block; width: 100px; font-weight: bold;}}
            input {{padding: 5px; border-radius: 4px; border: 1px solid #ddd; margin-right: 8px; width: 80px;}}
            button {{background: #2980b9; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 8px;}}
            button:hover {{background: #1f618d;}}
            #connected_devices, #real_power, #real_channel {{color: #e74c3c; font-weight: bold;}}
            .tip {{color: #666; font-size: 12px; margin-top: 5px;}}
            .config-section {{margin-bottom: 15px;}}
        </style>
    </head>
    <body>
        <h1>WiFi功率/信道配置面板</h1>
        
        <!-- 基础信息区 -->
        <div class="card">
            <h2>基础WiFi信息</h2>
            <p>WiFi名称(SSID)：{wifi_params['ssid']}</p>
            <p>无线接口：{wifi_params['interface']}</p>
            <p>当前信道：<span id="real_channel">{wifi_params['channel']}</span></p>
            <p>当前实际生效功率：<span id="real_power">{wifi_params['tx_power_actual']}</span> dBm</p>
            <p>已连接设备数：<span id="connected_devices">{connected_devices}</span></p>
            <p>当前工作模式：{wifi_params['mode']}</p>
        </div>
        
        <!-- 配置修改区 -->
        <div class="card">
            <h2>修改WiFi参数</h2>
            
            <!-- 信道修改表单 -->
            <div class="config-section">
                <div class="config-group">
                    <label>2.4G信道：</label>
                    <input type="number" name="channel" id="channel_input"
                           value="{wifi_params['channel']}" min="1" max="13" required>
                    <span class="tip">推荐：1/6/11（修改后会重启热点）</span>
                </div>
                <form action="/set_channel" method="post" style="display: inline;">
                    <input type="hidden" name="channel" id="channel_hidden" value="{wifi_params['channel']}">
                    <button type="submit" onclick="document.getElementById('channel_hidden').value = document.getElementById('channel_input').value">
                        保存信道并生效
                    </button>
                </form>
            </div>
            
            <!-- 功率修改表单 -->
            <div class="config-section">
                <div class="config-group">
                    <label>传输功率：</label>
                    <input type="number" name="tx_power" id="power_input"
                           value="{wifi_params['tx_power_actual']}" min="1" max="20" required>
                    <span>dBm</span>
                    <span class="tip">硬件可能自动修正数值（如4→3、16→15），修改后实时生效</span>
                </div>
                <form action="/set_power" method="post" style="display: inline;">
                    <input type="hidden" name="tx_power" id="power_hidden" value="{wifi_params['tx_power_actual']}">
                    <button type="submit" onclick="document.getElementById('power_hidden').value = document.getElementById('power_input').value">
                        保存功率并生效
                    </button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting combined Jetson server...")
    print(f"WiFi API: http://0.0.0.0:5000")
    print(f"Socket server: port 9000")
    app.run(host='0.0.0.0', port=5000, debug=False)
