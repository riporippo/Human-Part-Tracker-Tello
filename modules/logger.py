import csv
import cv2
import os
import socket
import struct
import numpy as np
from datetime import datetime

class ExperimentLogger:
    def __init__(self, host_ip, host_port=9999, base_dir="logs"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.log_dir, "flight_log.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        self.headers = [
            "timestamp", "mode", "battery", 
            "target_found", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence",
            "rc_lr", "rc_fb", "rc_ud", "rc_y",
            "ai_response","inference_latency_ms", "api_latency_ms",
            "error_yaw", "error_pitch", "error_throttle","loop_duration_ms"
        ]
        self.writer.writerow(self.headers)

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.use_network_video = False
        
        try:
            print(f"[Logger] Connecting to Host PC ({host_ip}:{host_port})...")
            self.client_socket.connect((host_ip, host_port))
            self.use_network_video = True
            print("[Logger] Video streaming connected!")
        except Exception as e:
            print(f"[Logger] Failed to connect to Host PC: {e}")

    def init_video(self, frame_width, frame_height, fps=30):
        pass

    def log_data(self, data_dict):
        row = [data_dict.get(h, "") for h in self.headers]
        self.writer.writerow(row)
        self.csv_file.flush()

    def write_frame(self, frame):
        if not self.use_network_video or frame is None:
            return
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
            if result:
                data = np.array(encoded_frame)
                string_data = data.tobytes()
                self.client_socket.sendall(struct.pack(">L", len(string_data)) + string_data)
        except Exception as e:
            print(f"[Logger] Send Error: {e}")
            self.use_network_video = False

    def close(self):
        if self.csv_file: self.csv_file.close()
        if self.use_network_video:
            self.client_socket.close()
        print(f"[Logger] Log saved to: {self.log_dir}")