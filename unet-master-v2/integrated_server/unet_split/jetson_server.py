# -*- coding: utf-8 -*-
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
from model_encoder import UNetEncoder

# Config
HOST = ''  # Listen on all interfaces
PORT = 9000
SAVE_DIR = '/home/nvidia/Pictures/images'  # Image save directory

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Global variables
server_socket = None
running = True

# 加载编码器（使用 CPU 模式避免 CUDA 兼容性问题）
device = torch.device('cpu')
encoder = UNetEncoder(in_channels=3).to(device)
encoder.load_state_dict(torch.load('saved_weights/encoder_weights.pth', map_location=device))
encoder.eval()
print(f"Encoder loaded on {device}")

# 预处理图像
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

# 编码图像
def encode_image(image_path):
    print("Encoding image...")
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        bottleneck, features = encoder(image_tensor)
    
    print(f"Encoding completed. Bottleneck shape: {bottleneck.shape}")
    return bottleneck, features

# 发送特征
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

def handle_client(conn, addr):
    """Handle client connection, receive image"""
    print(f"\n[+] New connection from {addr}")
    
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


def main():
    global server_socket, running
    
    # Create TCP server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"[*] Jetson Image Server started on port {PORT}")
        print(f"[*] Images will be saved to: {SAVE_DIR}")
        print("[*] Waiting for connections...")
        print("[*] Press Ctrl+C to stop\n")
        
        while running:
            try:
                # Set accept timeout for Ctrl+C response
                server_socket.settimeout(1.0)
                try:
                    conn, addr = server_socket.accept()
                    # Handle client in new thread
                    client_thread = threading.Thread(
                        target=handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                    
            except KeyboardInterrupt:
                print("\n[!] Shutting down server...")
                running = False
                break
            except Exception as e:
                print(f"[!] Server error: {e}")
                
    except Exception as e:
        print(f"[!] Failed to start server: {e}")
        sys.exit(1)
    finally:
        if server_socket:
            server_socket.close()
        print("[*] Server stopped")


if __name__ == '__main__':
    main()
