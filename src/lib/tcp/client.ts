// TCP客户端服务，用于与Jetson设备通信
export class TcpClient {
  private ip: string;
  private port: number;
  private socket: WebSocket | null = null;
  private messageHandlers: ((data: string) => void)[] = [];

  constructor(ip: string, port: number) {
    this.ip = ip;
    this.port = port;
  }

  // 连接到Jetson
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // 注意：这里使用WebSocket，因为浏览器不支持直接的TCP套接字
        // 在实际生产环境中，可能需要在服务器端建立一个代理
        // 这里我们使用WebSocket来模拟TCP通信
        // 注意：Jetson端需要运行一个WebSocket服务器，或者使用代理
        // 由于这是一个模拟实现，实际使用时需要根据Jetson端的实现进行调整
        
        // 模拟连接成功
        console.log(`Connecting to Jetson at ${this.ip}:${this.port}`);
        
        // 模拟延迟
        setTimeout(() => {
          console.log('Connected to Jetson');
          resolve();
        }, 500);
      } catch (error) {
        reject(error);
      }
    });
  }

  // 发送字符串数据
  async send(data: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log(`Sending data to Jetson: ${data}`);
        // 模拟发送成功
        setTimeout(() => {
          resolve();
        }, 100);
      } catch (error) {
        reject(error);
      }
    });
  }

  // 发送二进制数据
  async sendBuffer(buffer: ArrayBuffer): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log(`Sending binary data to Jetson (${buffer.byteLength} bytes)`);
        // 模拟发送成功
        setTimeout(() => {
          resolve();
        }, 1000); // 模拟较大数据的传输时间
      } catch (error) {
        reject(error);
      }
    });
  }

  // 断开连接
  async disconnect(): Promise<void> {
    return new Promise((resolve) => {
      console.log('Disconnecting from Jetson');
      // 模拟断开连接
      setTimeout(() => {
        console.log('Disconnected from Jetson');
        resolve();
      }, 100);
    });
  }

  // 注册消息处理器
  onMessage(handler: (data: string) => void): void {
    this.messageHandlers.push(handler);
  }

  // 移除消息处理器
  offMessage(handler: (data: string) => void): void {
    this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
  }
}
