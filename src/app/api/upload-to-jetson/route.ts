import { NextRequest, NextResponse } from 'next/server';
import net from 'net';

const JETSON_PORT = 9000;
const TIMEOUT = 30000;

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const formData = await request.formData();
    const file = formData.get('image') as File;
    const jetsonIp = formData.get('jetsonIp') as string;
    
    if (!file) {
      return NextResponse.json(
        { success: false, error: 'No image file provided' },
        { status: 400 }
      );
    }

    if (!jetsonIp) {
      return NextResponse.json(
        { success: false, error: 'No Jetson IP provided' },
        { status: 400 }
      );
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    console.log(`Preparing to upload ${file.name} (${buffer.length} bytes) to Jetson at ${jetsonIp}:${JETSON_PORT}`);

    const result = await sendToJetson(jetsonIp, file.name, buffer);

    if (result.success) {
      return NextResponse.json({
        success: true,
        message: 'Image uploaded successfully',
        filename: file.name,
        size: buffer.length,
        jetsonIp: jetsonIp,
      });
    } else {
      return NextResponse.json(
        { success: false, error: result.error },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}

function sendToJetson(jetsonIp: string, filename: string, data: Buffer): Promise<{ success: boolean; error?: string }> {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let isResolved = false;

    socket.setTimeout(TIMEOUT);

    socket.connect(JETSON_PORT, jetsonIp, () => {
      console.log(`Connected to Jetson at ${jetsonIp}`);

      const header = `IMAGE:${filename}:${data.length}\n`;
      socket.write(header, 'utf-8', () => {
        console.log('Header sent:', header.trim());
        
        setTimeout(() => {
          socket.write(data, () => {
            console.log('Image data sent');
            
            socket.write('\nEND_OF_IMAGE\n', () => {
              console.log('End marker sent');
              
              if (!isResolved) {
                isResolved = true;
                resolve({ success: true });
                socket.end();
              }
            });
          });
        }, 100);
      });
    });

    socket.on('error', (err) => {
      console.error('Socket error:', err.message);
      if (!isResolved) {
        isResolved = true;
        resolve({ success: false, error: `Connection error: ${err.message}` });
        socket.destroy();
      }
    });

    socket.on('timeout', () => {
      console.error('Socket timeout');
      if (!isResolved) {
        isResolved = true;
        resolve({ success: false, error: 'Connection timeout' });
        socket.destroy();
      }
    });

    socket.on('close', () => {
      console.log('Connection closed');
      if (!isResolved) {
        isResolved = true;
        resolve({ success: true });
      }
    });
  });
}
