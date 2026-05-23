# -*- coding: utf-8 -*-
import socket
import threading
import json
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edge_encoder import DIENEdgeEncoder

# 配置
HOST = ''  # 监听所有接口
PORT = 9001

# 全局变量
encoder = None

def handle_client(conn, addr):
    """处理客户端连接"""
    print(f"\n[+] 新连接来自 {addr}")
    
    try:
        # 设置超时
        conn.settimeout(60)
        
        # 接收数据缓冲区
        buffer = b''
        header_received = False
        expected_size = 0
        
        while True:
            try:
                # 接收数据
                chunk = conn.recv(4096)
                if not chunk:
                    break
                
                buffer += chunk
                
                # 解析头部信息
                if not header_received and b'\n' in buffer:
                    # 找到第一行（头部）
                    header_end = buffer.find(b'\n')
                    header = buffer[:header_end].decode('utf-8', errors='ignore').strip()
                    buffer = buffer[header_end + 1:]
                    
                    print(f"[*] 收到头部: {header}")
                    
                    # 解析头部格式: DATA:size
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
                
                # 检查数据是否完整
                if header_received and len(buffer) >= expected_size:
                    # 提取数据
                    data_bytes = buffer[:expected_size]
                    buffer = buffer[expected_size:]
                    
                    # 解析数据
                    data = json.loads(data_bytes.decode('utf-8'))
                    print(f"[*] 收到数据: {data}")
                    
                    # 提取用户历史行为
                    user_id = data.get('user_id')
                    item_id = data.get('item_id')
                    category_id = data.get('category_id')
                    history_items = data.get('history_items', [])
                    history_categories = data.get('history_categories', [])
                    
                    # 生成隐向量
                    try:
                        seq_length = len(history_items)
                        vector = encoder.encode(user_id, item_id, category_id, history_items, history_categories, seq_length)
                        
                        # 构建响应
                        response = {
                            'status': 'success',
                            'vector': vector,
                            'message': '编码成功'
                        }
                    except Exception as e:
                        print(f"[!] 编码失败: {e}")
                        response = {
                            'status': 'error',
                            'message': str(e)
                        }
                    
                    # 发送响应
                    response_bytes = json.dumps(response).encode('utf-8')
                    response_header = f"DATA:{len(response_bytes)}"
                    conn.sendall(response_header.encode('utf-8') + b'\n')
                    conn.sendall(response_bytes)
                    print(f"[+] 响应已发送")
                    
                    # 处理完一个请求后关闭连接
                    break
                    
                # 防止缓冲区溢出
                if len(buffer) > 10 * 1024 * 1024:  # 10MB限制
                    print("[!] 缓冲区过大，清空...")
                    buffer = b''
                    header_received = False
                    expected_size = 0
                    
            except socket.timeout:
                print("[!] 接收超时")
                break
            except Exception as e:
                print(f"[!] 接收数据错误: {e}")
                break
                
    except Exception as e:
        print(f"[!] 客户端处理错误: {e}")
    finally:
        conn.close()
        print(f"[-] 连接关闭: {addr}")

def main():
    """主函数"""
    global encoder
    
    # 加载编码器
    model_path = 'checkpoints/ckpt_noshuffDIEN3_d64'
    vocab_path = 'data'
    
    try:
        encoder = DIENEdgeEncoder(model_path, vocab_path)
        print("编码器加载成功")
    except Exception as e:
        print(f"编码器加载失败: {e}")
        sys.exit(1)
    
    # 创建TCP服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"[*] 边缘端编码器服务启动，监听端口{PORT}...")
        print("[*] 等待连接...")
        print("[*] 按Ctrl+C停止")
        
        running = True
        while running:
            try:
                # 设置accept超时，以便响应Ctrl+C
                server_socket.settimeout(1.0)
                try:
                    conn, addr = server_socket.accept()
                    # 在新线程中处理客户端
                    client_thread = threading.Thread(
                        target=handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                    
            except KeyboardInterrupt:
                print("\n[!] 正在关闭服务器...")
                running = False
                break
            except Exception as e:
                print(f"[!] 服务器错误: {e}")
                
    except Exception as e:
        print(f"[!] 服务启动失败: {e}")
        return
    finally:
        if server_socket:
            server_socket.close()
        print("[*] 服务器已停止")

if __name__ == '__main__':
    main()
