# -*- coding: utf-8 -*-
import socket
import threading
import json
import os
import sys
import time
from datetime import datetime
import subprocess

# ==================== 服务器配置 ====================
HOST = ''  # 监听所有接口
HTTP_PORT = 5001  # HTTP服务器端口（用于WiFi设置）

# ==================== WiFi配置 ====================
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
    return 20

def get_connected_devices():
    """获取连接的设备数"""
    try:
        result = subprocess.run(['iw', 'dev', WIFI_INTERFACE, 'station', 'dump'], 
                                capture_output=True, text=True)
        if result.returncode == 0:
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
            try:
                channel = get_wifi_channel()
                self._set_response('application/json')
                self.wfile.write(json.dumps({'channel': str(channel)}).encode('utf-8'))
            except Exception as e:
                print(f"[!] 获取WiFi信道错误: {e}")
                self._set_response('application/json')
                self.wfile.write(json.dumps({'channel': '6'}).encode('utf-8'))
        
        elif self.path == '/get_real_power':
            try:
                power = get_wifi_power()
                self._set_response('application/json')
                self.wfile.write(json.dumps({'power': str(power)}).encode('utf-8'))
            except Exception as e:
                print(f"[!] 获取WiFi功率错误: {e}")
                self._set_response('application/json')
                self.wfile.write(json.dumps({'power': '20'}).encode('utf-8'))
        
        elif self.path == '/get_connected_devices':
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
        
        try:
            post_data = json.loads(post_data.decode('utf-8'))
            print(f"[+] HTTP POST请求: {self.path}, JSON数据: {post_data}")
        except json.JSONDecodeError:
            import urllib.parse
            post_data = urllib.parse.parse_qs(post_data.decode('utf-8'))
            post_data = {k: v[0] for k, v in post_data.items()}
            print(f"[+] HTTP POST请求: {self.path}, 表单数据: {post_data}")
        
        if self.path == '/set_wifi_channel' or self.path == '/set_channel':
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
    print(f"  - 设置信道: POST http://{HOST}:{HTTP_PORT}/set_channel")
    print(f"  - 设置功率: POST http://{HOST}:{HTTP_PORT}/set_power")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("HTTP服务器关闭")
        httpd.shutdown()

if __name__ == '__main__':
    print("=" * 50)
    print("WiFi边缘端服务器启动")
    print("=" * 50)
    print(f"HTTP服务器（WiFi设置）: 端口 {HTTP_PORT}")
    print("=" * 50)

    # 启动HTTP服务器线程
    http_thread = threading.Thread(target=start_http_server)
    http_thread.daemon = True
    http_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器关闭")
