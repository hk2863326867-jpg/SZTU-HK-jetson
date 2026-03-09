# -*- coding: utf-8 -*-
import socket
import sys
import os
import threading
import time
from datetime import datetime

# Config
HOST = ''  # Listen on all interfaces
PORT = 9000
SAVE_DIR = '/home/nvidia/Pictures/images'  # Image save directory, change to your actual path

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Global variables
server_socket = None
running = True


def handle_client(conn, addr):
    """Handle client connection, receive image"""
    print(f"\n[+] New connection from {addr}")
    
    try:
        # Set timeout
        conn.settimeout(30)
        
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
                        
                        # Send confirmation
                        conn.send(f"OK:Image received and saved as {safe_filename}\n".encode('utf-8'))
                    
                    # Clear buffer for next image
                    buffer = b''
                    header_received = False
                    filename = None
                    expected_size = 0
                    
                # Prevent buffer overflow
                if len(buffer) > 10 * 1024 * 1024:  # 10MB limit
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
        print(f"[*] Image server started on port {PORT}")
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
