import socket
import cv2
import numpy as np
import struct
import os
from datetime import datetime
import dotenv

dotenv.load_dotenv()

HOST_IP = "0.0.0.0"
PORT = int(os.getenv("HOST_PC_PORT", 9999))
SAVE_DIR = "received_logs"

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST_IP, PORT))
    server_socket.listen(1)
    print(f"[Debug] Waiting for connection on port {PORT}...")

    conn, addr = server_socket.accept()
    print(f"[Debug] Connected by {addr}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(SAVE_DIR, f"flight_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    try:
        data = b""
        payload_size = struct.calcsize(">L") 

        while True:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet
            if not data: break
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                    print(f"[Debug] Recording started: {video_path}")

                out.write(frame)
                cv2.imshow('Host PC Receiver', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"[Debug] Error: {e}")
    finally:
        conn.close()
        server_socket.close()
        if out: out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()