# -*- coding: utf-8 -*-
import socket
import threading
import json
import os
import sys
import time
from datetime import datetime
import torch
import numpy as np
import skimage.io as io
import skimage.transform as trans

# ==================== 添加模块路径 ====================
# 添加UNet模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'unet_split'))
from unet_split.model_encoder import UNetEncoder

# 添加DIEN模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'dien_split'))
from dien_split.edge_encoder import DIENEdgeEncoder

# ==================== 服务器配置 ====================
HOST = ''  # 监听所有接口
UNET_PORT = 9000  # UNet图像分割端口
DIEN_PORT = 9001  # DIEN推荐端口
HTTP_PORT = 5000  # HTTP服务器端口（用于WiFi设置）

# ==================== 全局变量 ====================
unet_encoder = None
dien_encoder = None

# ==================== UNet 编码器函数 ====================
def preprocess_image(image_path):
    """预处理图像"""
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
    """编码图像"""
    print("Encoding image...")
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        bottleneck, features = unet_encoder(image_tensor)

    print(f"Encoding completed. Bottleneck shape: {bottleneck.shape}")
    return bottleneck, features

def send_features(conn, bottleneck, features):
    """发送特征"""
    print("Sending features...")

    bottleneck_np = bottleneck.cpu().numpy()
    features_np = [f.cpu().numpy() for f in features]

    # 发送特征数量
    num_features = len(features_np)
    conn.sendall(num_features.to_bytes(4, byteorder='little'))

    # 发送瓶颈层特征
    bottleneck_shape = bottleneck_np.shape
    for dim in bottleneck_shape:
        conn.sendall(dim.to_bytes(4, byteorder='little'))
    conn.sendall(bottleneck_np.tobytes())

    # 发送跳跃连接特征
    for i, feature in enumerate(features_np):
        feature_shape = feature.shape
        for dim in feature_shape:
            conn.sendall(dim.to_bytes(4, byteorder='little'))
        conn.sendall(feature.tobytes())
        print(f"Sent feature {i+1}: {feature_shape}")

    print("All features sent successfully!")

def handle_unet_client(conn, addr):
    """处理UNet客户端连接"""
    print(f"\n[+] UNet新连接来自 {addr}")

    try:
        conn.settimeout(60)

        buffer = b''
        expected_size = 0
        filename = None
        header_received = False

        while True:
            try:
                chunk = conn.recv(4096)
                if not chunk:
                    break

                buffer += chunk

                if not header_received and b'\n' in buffer:
                    header_end = buffer.find(b'\n')
                    header = buffer[:header_end].decode('utf-8', errors='ignore').strip()
                    buffer = buffer[header_end + 1:]

                    print(f"[*] UNet收到头部: {header}")

                    if header.startswith('IMAGE:'):
                        parts = header.split(':')
                        if len(parts) >= 3:
                            filename = parts[1]
                            try:
                                expected_size = int(parts[2])
                                header_received = True
                                print(f"[*] 期望图片: {filename} ({expected_size} bytes)")
                            except ValueError:
                                print(f"[!] 头部大小无效: {parts[2]}")
                                break
                        else:
                            print(f"[!] 头部格式无效")
                            break
                    else:
                        print(f"[!] 未知头部: {header}")
                        break

                if header_received and b'END_OF_IMAGE' in buffer:
                    end_marker_pos = buffer.find(b'END_OF_IMAGE')
                    image_data = buffer[:end_marker_pos]

                    if filename and len(image_data) > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_filename = f"{timestamp}_{filename}"
                        filepath = os.path.join(SAVE_DIR, safe_filename)

                        with open(filepath, 'wb') as f:
                            f.write(image_data)

                        print(f"[+] 图片保存: {filepath}")

                        try:
                            bottleneck, features = encode_image(filepath)
                            send_features(conn, bottleneck, features)
                            conn.send(f"OK:Image processed and features sent\n".encode('utf-8'))
                        except Exception as e:
                            print(f"[!] 图片处理错误: {e}")
                            conn.send(f"ERROR:Failed to process image: {e}\n".encode('utf-8'))

                    buffer = b''
                    header_received = False
                    filename = None
                    expected_size = 0

                if len(buffer) > 20 * 1024 * 1024:
                    print("[!] 缓冲区过大，清空...")
                    buffer = b''
                    header_received = False

            except socket.timeout:
                print("[!] UNet接收超时")
                break
            except Exception as e:
                print(f"[!] UNet接收数据错误: {e}")
                break

    except Exception as e:
        print(f"[!] UNet客户端处理错误: {e}")
    finally:
        conn.close()
        print(f"[-] UNet连接关闭: {addr}")

def handle_dien_client(conn, addr):
    """处理DIEN客户端连接"""
    print(f"\n[+] DIEN新连接来自 {addr}")

    try:
        conn.settimeout(60)

        buffer = b''
        header_received = False
        expected_size = 0

        while True:
            try:
                chunk = conn.recv(4096)
                if not chunk:
                    break

                buffer += chunk

                if not header_received and b'\n' in buffer:
                    header_end = buffer.find(b'\n')
                    header = buffer[:header_end].decode('utf-8', errors='ignore').strip()
                    buffer = buffer[header_end + 1:]

                    print(f"[*] DIEN收到头部: {header}")

                    if header.startswith('DATA:'):
                        parts = header.split(':')
                        if len(parts) >= 2:
                            try:
                                expected_size = int(parts[1])
                                header_received = True
                                print(f"[*] 期望数据大小: {expected_size} bytes")
                            except ValueError:
                                print(f"[!] 头部大小无效: {parts[1]}")
                                break
                        else:
                            print(f"[!] 头部格式无效")
                            break
                    else:
                        print(f"[!] 未知头部: {header}")
                        break

                if header_received and len(buffer) >= expected_size:
                    data_bytes = buffer[:expected_size]
                    buffer = buffer[expected_size:]

                    data = json.loads(data_bytes.decode('utf-8'))
                    print(f"[*] DIEN收到数据: {data}")

                    user_id = data.get('user_id')
                    item_id = data.get('item_id')
                    category_id = data.get('category_id')
                    history_items = data.get('history_items', [])
                    history_categories = data.get('history_categories', [])

                    try:
                        seq_length = len(history_items)
                        vector = dien_encoder.encode(user_id, item_id, category_id, history_items, history_categories, seq_length)

                        response = {
                            'status': 'success',
                            'vector': vector,
                            'message': '编码成功'
                        }
                    except Exception as e:
                        print(f"[!] DIEN编码失败: {e}")
                        response = {
                            'status': 'error',
                            'message': str(e)
                        }

                    response_bytes = json.dumps(response).encode('utf-8')
                    response_header = f"DATA:{len(response_bytes)}"
                    conn.sendall(response_header.encode('utf-8') + b'\n')
                    conn.sendall(response_bytes)
                    print(f"[+] DIEN响应已发送")

                    break

                if len(buffer) > 10 * 1024 * 1024:
                    print("[!] DIEN缓冲区过大，清空...")
                    buffer = b''
                    header_received = False

            except socket.timeout:
                print("[!] DIEN接收超时")
                break
            except Exception as e:
                print(f"[!] DIEN接收数据错误: {e}")
                break

    except Exception as e:
        print(f"[!] DIEN客户端处理错误: {e}")
    finally:
        conn.close()
        print(f"[-] DIEN连接关闭: {addr}")

def start_unet_server():
    """启动UNet服务器"""
    unet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    unet_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    unet_socket.bind((HOST, UNET_PORT))
    unet_socket.listen(5)
    print(f"UNet边缘端服务器启动，监听端口 {UNET_PORT}")

    while True:
        conn, addr = unet_socket.accept()
        client_thread = threading.Thread(target=handle_unet_client, args=(conn, addr))
        client_thread.daemon = True
        client_thread.start()

def start_dien_server():
    """启动DIEN服务器"""
    dien_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dien_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    dien_socket.bind((HOST, DIEN_PORT))
    dien_socket.listen(5)
    print(f"DIEN边缘端服务器启动，监听端口 {DIEN_PORT}")

    while True:
        conn, addr = dien_socket.accept()
        client_thread = threading.Thread(target=handle_dien_client, args=(conn, addr))
        client_thread.daemon = True
        client_thread.start()

# ==================== WiFi工具函数 ====================
import subprocess

WIFI_INTERFACE = 'wlP1p1s0'  # WiFi接口名称，根据实际情况修改

def get_wifi_channel():
    """获取实际WiFi信道"""
    try:
        # 使用iw命令获取当前信道
        result = subprocess.run(['iw', 'dev', WIFI_INTERFACE, 'info'], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                # 查找包含 'channel' 的行，格式可能是：
                # channel 6 [2437 MHz] (width: 20 MHz, center1: 2437 MHz)
                if 'channel' in line and 'MHz' in line:
                    # 提取信道号（channel后面的数字）
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if part == 'channel' and i + 1 < len(parts):
                            try:
                                channel = int(parts[i + 1])
                                print(f"[WiFi] 从系统获取到信道: {channel}")
                                return channel
                            except ValueError:
                                continue
                # 也可能是更简单的格式：channel 6
                elif 'channel' in line and not 'MHz' in line:
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        try:
                            channel = int(parts[1].strip())
                            print(f"[WiFi] 从系统获取到信道: {channel}")
                            return channel
                        except ValueError:
                            continue
    except Exception as e:
        print(f"[WiFi] 获取信道命令执行失败: {e}")
    
    # 如果获取失败，返回默认值
    return 6

def get_wifi_power():
    """获取实际WiFi功率"""
    try:
        # 使用iw命令获取当前功率
        result = subprocess.run(['iw', 'dev', WIFI_INTERFACE, 'info'], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'txpower' in line.lower():
                    # 格式如: txpower 20.00 dBm
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        power = parts[1]
                        print(f"[WiFi] 从系统获取到功率: {power} dBm")
                        return int(float(power))
        # 如果iw命令没有返回功率，尝试另一种方式
        result = subprocess.run(['iwconfig', WIFI_INTERFACE], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Tx-Power' in line:
                    parts = line.split('=')[1].split(' ')
                    power = parts[0]
                    print(f"[WiFi] 从iwconfig获取到功率: {power} dBm")
                    return int(power)
    except Exception as e:
        print(f"[WiFi] 获取功率命令执行失败: {e}")
    
    # 如果获取失败，返回默认值
    return 20

def get_connected_devices():
    """获取连接的设备数"""
    try:
        # 使用iw命令获取连接的客户端
        result = subprocess.run(['iw', 'dev', WIFI_INTERFACE, 'station', 'dump'], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            # 计算包含"Station"的行数
            stations = result.stdout.count('Station')
            print(f"[WiFi] 从系统获取到连接设备数: {stations}")
            return stations
    except Exception as e:
        print(f"[WiFi] 获取连接设备数命令执行失败: {e}")
    
    return 0

def set_wifi_channel(channel):
    """设置WiFi信道 - 使用nmcli命令"""
    try:
        print(f"[WiFi] 正在设置信道: {channel}")
        
        # 修改信道
        result = subprocess.run(['sudo', 'nmcli', 'connection', 'modify', 'Board_Hotspot', 
                                f'802-11-wireless.channel', str(channel)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WiFi] 修改信道配置失败: {result.stderr}")
            return False, f"修改信道配置失败: {result.stderr}"
        
        # 重启热点
        subprocess.run(['sudo', 'nmcli', 'connection', 'down', 'Board_Hotspot'],
                        capture_output=True)
        subprocess.run(['sudo', 'nmcli', 'connection', 'up', 'Board_Hotspot'],
                        capture_output=True)
        
        print(f"[WiFi] 信道设置成功: {channel}")
        return True, f"信道已设置为 {channel}"
    except Exception as e:
        print(f"[WiFi] 设置信道失败: {e}")
        return False, str(e)

def set_wifi_power(power):
    """设置WiFi功率 - 使用iw命令"""
    try:
        print(f"[WiFi] 正在设置功率: {power} dBm")
        
        # 从测试结果看：实际功率 ≈ 输入值 / 100
        # 例如：设置 100 → 实际 1，设置 300 → 实际 3
        # 所以需要把 dBm 值乘以 100
        power_iw = int(power * 100)
        
        # 使用iw命令设置功率
        result = subprocess.run(['sudo', 'iw', 'dev', WIFI_INTERFACE, 
                                'set', 'txpower', 'fixed', str(power_iw)],
                                capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[WiFi] 功率设置成功: {power} dBm (使用 {power_iw})")
            return True, f"功率已设置为 {power} dBm"
        else:
            error_msg = result.stderr or "未知错误"
            print(f"[WiFi] 功率设置失败: {error_msg}")
            return False, f"功率设置失败: {error_msg}"
    except Exception as e:
        print(f"[WiFi] 设置功率失败: {e}")
        return False, str(e)

# ==================== HTTP服务器（用于WiFi设置） ====================
import http.server
import socketserver
import json

class WiFiHTTPHandler(http.server.BaseHTTPRequestHandler):
    """处理WiFi相关的HTTP请求"""
    
    def _set_response(self, content_type='text/plain'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
    
    def do_GET(self):
        """处理GET请求"""
        print(f"[+] HTTP GET请求: {self.path}")
        
        if self.path == '/get_real_channel':
            # 获取实际WiFi信道
            try:
                channel = get_wifi_channel()
                self._set_response('application/json')
                self.wfile.write(json.dumps({'channel': str(channel)}).encode('utf-8'))
            except Exception as e:
                print(f"[!] 获取WiFi信道错误: {e}")
                self._set_response('application/json')
                self.wfile.write(json.dumps({'channel': '6'}).encode('utf-8'))
        
        elif self.path == '/get_real_power':
            # 获取实际WiFi功率
            try:
                power = get_wifi_power()
                self._set_response('application/json')
                self.wfile.write(json.dumps({'power': str(power)}).encode('utf-8'))
            except Exception as e:
                print(f"[!] 获取WiFi功率错误: {e}")
                self._set_response('application/json')
                self.wfile.write(json.dumps({'power': '20'}).encode('utf-8'))
        
        elif self.path == '/get_connected_devices':
            # 获取连接的设备数
            try:
                devices = get_connected_devices()
                print(f"[*] 获取连接设备数: {devices}")
                self._set_response('text/plain')
                self.wfile.write(str(devices).encode('utf-8'))
            except Exception as e:
                print(f"[!] 获取连接设备数错误: {e}")
                self._set_response('text/plain')
                self.wfile.write('0'.encode('utf-8'))
        
        else:
            self._set_response()
            self.wfile.write(b'WiFi HTTP Server')
    
    def do_POST(self):
        """处理POST请求"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # 尝试解析JSON数据，如果失败则尝试解析表单数据
        try:
            post_data = json.loads(post_data.decode('utf-8'))
            print(f"[+] HTTP POST请求: {self.path}, JSON数据: {post_data}")
        except json.JSONDecodeError:
            # 尝试解析表单数据
            import urllib.parse
            post_data = urllib.parse.parse_qs(post_data.decode('utf-8'))
            # 将表单数据转换为字典
            post_data = {k: v[0] for k, v in post_data.items()}
            print(f"[+] HTTP POST请求: {self.path}, 表单数据: {post_data}")
        
        if self.path == '/set_wifi_channel' or self.path == '/set_channel':
            # 设置WiFi信道
            try:
                channel = post_data.get('channel')
                if channel is None:
                    raise ValueError("缺少channel参数")
                print(f"[*] 设置WiFi信道: {channel}")
                
                success, message = set_wifi_channel(int(channel))
                
                self._set_response('application/json')
                self.wfile.write(json.dumps({'success': success, 'message': message}).encode('utf-8'))
            except Exception as e:
                print(f"[!] 设置WiFi信道错误: {e}")
                self._set_response('application/json')
                self.wfile.write(json.dumps({'success': False, 'message': str(e)}).encode('utf-8'))
        
        elif self.path == '/set_wifi_power' or self.path == '/set_power':
            # 设置WiFi功率
            try:
                power = post_data.get('power') or post_data.get('tx_power')
                if power is None:
                    raise ValueError("缺少power参数")
                print(f"[*] 设置WiFi功率: {power}")
                
                success, message = set_wifi_power(int(power))
                
                self._set_response('application/json')
                self.wfile.write(json.dumps({'success': success, 'message': message}).encode('utf-8'))
            except Exception as e:
                print(f"[!] 设置WiFi功率错误: {e}")
                self._set_response('application/json')
                self.wfile.write(json.dumps({'success': False, 'message': str(e)}).encode('utf-8'))
        
        else:
            self._set_response()
            self.wfile.write(b'WiFi HTTP Server')

def start_http_server():
    """启动HTTP服务器"""
    handler = WiFiHTTPHandler
    httpd = socketserver.TCPServer((HOST, HTTP_PORT), handler)
    print(f"HTTP服务器启动，监听端口 {HTTP_PORT}")
    print(f"WiFi API端点:")
    print(f"  - 获取信道: http://{HOST}:{HTTP_PORT}/get_real_channel")
    print(f"  - 获取功率: http://{HOST}:{HTTP_PORT}/get_real_power")
    print(f"  - 获取连接设备数: http://{HOST}:{HTTP_PORT}/get_connected_devices")
    print(f"  - 设置信道: POST http://{HOST}:{HTTP_PORT}/set_wifi_channel")
    print(f"  - 设置功率: POST http://{HOST}:{HTTP_PORT}/set_wifi_power")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("HTTP服务器关闭")
        httpd.shutdown()

def load_models():
    """加载所有模型"""
    global unet_encoder, dien_encoder, device

    # 加载UNet编码器
    try:
        device = torch.device('cpu')
        unet_encoder = UNetEncoder(in_channels=3).to(device)
        unet_encoder.load_state_dict(torch.load('unet_split/saved_weights/saved_weights/encoder_weights.pth', map_location=device))
        unet_encoder.eval()
        print(f"UNet编码器加载成功，运行在 {device}")
    except Exception as e:
        print(f"UNet编码器加载失败: {e}")

    # 加载DIEN编码器
    try:
        # 注意：实际路径多了一层checkpoints目录
        dien_model_path = 'dien_split/checkpoints/checkpoints/ckpt_noshuffDIEN3_d64'
        dien_vocab_path = 'dien_split/data/data'
        dien_encoder = DIENEdgeEncoder(dien_model_path, dien_vocab_path)
        print("DIEN编码器加载成功")
    except Exception as e:
        print(f"DIEN编码器加载失败: {e}")

if __name__ == '__main__':
    SAVE_DIR = 'images'
    os.makedirs(SAVE_DIR, exist_ok=True)

    load_models()

    print("=" * 50)
    print("整合边缘端服务器启动")
    print("=" * 50)
    print(f"UNet图像分割服务: 端口 {UNET_PORT}")
    print(f"DIEN推荐编码服务: 端口 {DIEN_PORT}")
    print(f"HTTP服务器（WiFi设置）: 端口 {HTTP_PORT}")
    print("=" * 50)

    # 启动三个服务器线程
    unet_thread = threading.Thread(target=start_unet_server)
    dien_thread = threading.Thread(target=start_dien_server)
    http_thread = threading.Thread(target=start_http_server)

    unet_thread.daemon = True
    dien_thread.daemon = True
    http_thread.daemon = True

    unet_thread.start()
    dien_thread.start()
    http_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器关闭")