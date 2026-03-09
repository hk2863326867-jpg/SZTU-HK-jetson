import socket
import sys
import threading
import time

# 全局变量：控制线程退出、标记连接状态
exit_flag = False
client_socket = None

# 接收消息线程（独立运行，实时收消息）
def recv_thread():
    global exit_flag, client_socket
    while not exit_flag:
        try:
            if client_socket:
                # 非阻塞接收
                client_socket.setblocking(False)
                try:
                    recv_data = client_socket.recv(1024).decode('utf-8', errors='ignore').strip()
                    if recv_data:
                        if recv_data.lower() == 'exit':
                            print("\nJetson send exit command, closing connection...")
                            exit_flag = True
                            break
                        print("\n[Jetson] > " + recv_data)
                        # 提示继续输入
                        print("[PC] > ", end='', flush=True)
                except BlockingIOError:
                    pass
                except ConnectionResetError:
                    print("\nJetson disconnected unexpectedly")
                    exit_flag = True
                    break
                client_socket.setblocking(True)
        except Exception as e:
            print("\nRecv error: " + str(e))
            exit_flag = True
            break
        time.sleep(0.1)

# 发送消息线程（独立运行，随时发消息）
def send_thread():
    global exit_flag, client_socket
    while not exit_flag:
        try:
            if client_socket:
                print("[PC] > ", end='', flush=True)
                send_data = sys.stdin.readline().strip()
                if send_data.lower() == 'exit':
                    client_socket.send("exit".encode('utf-8'))
                    print("Send exit command, closing connection...")
                    exit_flag = True
                    break
                if send_data:  # 非空才发送
                    client_socket.send(send_data.encode('utf-8'))
        except Exception as e:
            print("\nSend error: " + str(e))
            exit_flag = True
            break
        time.sleep(0.05)

def main():
    global exit_flag, client_socket
    JETSON_IP = '192.168.55.1'  # 替换为你的Jetson IP
    PORT = 9000

    # 创建客户端套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((JETSON_IP, PORT))
        print("Connected to Jetson: " + JETSON_IP + ":" + str(PORT))
        print("===== Free chat mode (type 'exit' to quit) =====")
    except Exception as e:
        print("Connection failed: " + str(e))
        print("Check: 1.Same LAN 2.Jetson server running 3.IP/Port correct 4.Firewall closed")
        sys.exit(1)

    # 启动收发线程（异步运行）
    t_recv = threading.Thread(target=recv_thread, daemon=True)
    t_send = threading.Thread(target=send_thread, daemon=True)
    t_recv.start()
    t_send.start()

    # 主线程等待退出
    while not exit_flag:
        time.sleep(0.1)

    # 清理资源1
    
    client_socket.close()
    print("Connection closed")

if __name__ == '__main__':
    main()