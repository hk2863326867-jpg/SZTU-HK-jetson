import socket
import sys
import threading
import time

# 全局变量：控制线程退出、标记连接状态
exit_flag = False
conn = None

# 接收消息线程（独立运行，实时收消息）
def recv_thread():
    global exit_flag, conn
    while not exit_flag:
        try:
            if conn:
                # 非阻塞接收（避免卡死）
                conn.setblocking(False)
                try:
                    recv_data = conn.recv(1024).decode('utf-8', errors='ignore').strip()
                    if recv_data:
                        if recv_data.lower() == 'exit':
                            print("\nPC send exit command, closing connection...")
                            exit_flag = True
                            break
                        print("\n[PC] > " + recv_data)
                        # 提示继续输入（不阻塞）
                        print("[Jetson] > ", end='', flush=True)
                except BlockingIOError:
                    pass  # 无数据时跳过
                except ConnectionResetError:
                    print("\nPC disconnected unexpectedly")
                    exit_flag = True
                    break
                conn.setblocking(True)
        except Exception as e:
            print("\nRecv error: " + str(e))
            exit_flag = True
            break
        time.sleep(0.1)  # 降低CPU占用

# 发送消息线程（独立运行，随时发消息）
def send_thread():
    global exit_flag, conn
    while not exit_flag:
        try:
            if conn:
                print("[Jetson] > ", end='', flush=True)
                send_data = sys.stdin.readline().strip()
                if send_data.lower() == 'exit':
                    conn.send("exit".encode('utf-8'))
                    print("Send exit command, closing connection...")
                    exit_flag = True
                    break
                if send_data:  # 非空才发送
                    conn.send(send_data.encode('utf-8'))
        except Exception as e:
            print("\nSend error: " + str(e))
            exit_flag = True
            break
        time.sleep(0.05)

def main():
    global exit_flag, conn
    HOST = ''
    PORT = 9000

    # 创建服务端套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((HOST, PORT))
    except Exception as e:
        print("Port bind failed: " + str(e))
        sys.exit(1)
    server_socket.listen(1)
    print("Jetson server started, listening on port " + str(PORT) + "...")

    # 等待客户端连接
    try:
        conn, addr = server_socket.accept()
        print("Connected to PC: " + str(addr))
        print("===== Free chat mode (type 'exit' to quit) =====")
    except KeyboardInterrupt:
        server_socket.close()
        print("\nServer closed")
        sys.exit(0)

    # 启动收发线程（异步运行）
    t_recv = threading.Thread(target=recv_thread, daemon=True)
    t_send = threading.Thread(target=send_thread, daemon=True)
    t_recv.start()
    t_send.start()

    # 主线程等待退出
    while not exit_flag:
        time.sleep(0.1)

    # 清理资源
    if conn:
        conn.close()
    server_socket.close()
    print("Connection closed")

if __name__ == '__main__':
    main()