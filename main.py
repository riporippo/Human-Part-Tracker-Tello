import time
import cv2
import threading
from djitellopy import Tello
from modules.logger import ExperimentLogger
from modules.tracker import PIDTracker
from modules.detectors import HailoPoseDetector
import sys
import os 
import dotenv

dotenv.load_dotenv()

HOST_PC_IP = os.getenv("HOST_PC_IP")

# COCO キーポイントのマッピング
KEYPOINT_DICT = {
    0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
    5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
    9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
    13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
}

# CUI入力処理スレッド
class InputThread(threading.Thread):
    def __init__(self, stop_event, command_queue):
        super().__init__()
        self.stop_event = stop_event
        self.command_queue = command_queue
        self.daemon = True

    def run(self):
        print("\n[Debug] === Command List ===")
        print(" [システム] t: 離陸 | l: 着陸 | q: 終了")
        print(" [モード]   auto: 自律追跡開始 | manual: 手動操作に戻す")
        print(" [手動移動] w/s: 前後 | a/d: 左右 | up/down: 上下 | cw/ccw: 旋回 | x: 停止(ホバリング)")
        print("============================\n")
        
        while not self.stop_event.is_set():
            try:
                command = input()
                if command:
                    self.command_queue.append(command.strip().lower())
            except EOFError:
                break
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Debug] Input thread error: {e}")
                break
# ----------------------------------------------------

def main():
    print("--- Setup Target Body Part ---")
    for k, v in KEYPOINT_DICT.items():
        print(f" {k}: {v}")
    
    target_kp_idx = -1
    while target_kp_idx < 0 or target_kp_idx > 16:
        try:
            val = input("Target Keypoint ID (0-16) [Default: 0 (Nose)]: ")
            if val == "":
                target_kp_idx = 0
            else:
                target_kp_idx = int(val)
        except ValueError:
            print("[Debug] Invalid input. Please enter a number between 0 and 16.")
    
    print(f"[Debug] Selected Target Keypoint: {target_kp_idx} ({KEYPOINT_DICT[target_kp_idx]})")

    tello = None
    logger = None
    command_queue = [] 
    stop_event = threading.Event()
    input_thread = None

    try:
        print("[Debug] Tello Real Connection Start")
        tello = Tello()
        tello.connect()
        tello.streamon()
        print(f"[Debug] Tello Connected! Battery: {tello.get_battery()}%")

        print("[Debug] Experiment Mode: hailo_pose")
        logger = ExperimentLogger(host_ip=HOST_PC_IP)
        
        frame_read = tello.get_frame_read()
        time.sleep(2) 
        
        first_frame = frame_read.frame
        if first_frame is None:
            raise RuntimeError("[Error] Failed to receive video stream from Tello.")

        h, w, _ = first_frame.shape
        logger.init_video(w, h)
        
        detector = HailoPoseDetector('models/yolov8m_pose.hef')
        tracker = PIDTracker() 

        # 状態管理フラグ
        is_flying = False
        is_tracking = False  # True: 自律追跡, False: 手動操作
        manual_rc_cmd = [0, 0, 0, 0] # 手動操作時のRCコマンド保持用
        
        input_thread = InputThread(stop_event, command_queue)
        input_thread.start()

        while not stop_event.is_set():
            loop_start_time = time.perf_counter()
            frame = frame_read.frame
            if frame is None: continue

            log_data = {
                "timestamp": time.time(), "mode": "hailo_pose", "battery": tello.get_battery(),
                "target_found": False, "rc_lr": 0, "rc_fb": 0, "rc_ud": 0, "rc_y": 0,
                "error_yaw": 0.0, "error_pitch": 0.0, "error_throttle": 0.0,
                "loop_duration_ms": 0.0
            }
            
            rc_cmd = [0, 0, 0, 0]

            # CUIコマンド処理
            if command_queue:
                command = command_queue.pop(0) 
                
                # --- システムコマンド ---
                if command == 't' and not is_flying:
                    tello.takeoff()
                    is_flying = True
                    is_tracking = False # 離陸時は安全のため必ず手動モード
                    manual_rc_cmd = [0, 0, 0, 0]
                    print("[Debug] TAKEOFF - Entered Manual Mode")
                    
                elif command == 'q':
                    stop_event.set() 
                    break
                    
                elif command == 'l' and is_flying:
                    tello.land()
                    is_flying = False
                    is_tracking = False
                    print("[Debug] LANDED")
                
                # --- モード切替コマンド ---
                elif command == 'auto':
                    is_tracking = True
                    print("[Debug] >>> AUTO TRACKING STARTED <<<")
                    
                elif command == 'manual':
                    is_tracking = False
                    manual_rc_cmd = [0, 0, 0, 0] # 切り替え時は安全のためホバリング
                    print("[Debug] >>> SWITCHED TO MANUAL MODE <<<")
                
                # --- 手動移動コマンド (手動モード時のみ受付) ---
                elif not is_tracking and is_flying:
                    speed = 30 # 手動移動時のスピード
                    if command == 'w': manual_rc_cmd = [0, speed, 0, 0]
                    elif command == 's': manual_rc_cmd = [0, -speed, 0, 0]
                    elif command == 'a': manual_rc_cmd = [-speed, 0, 0, 0]
                    elif command == 'd': manual_rc_cmd = [speed, 0, 0, 0]
                    elif command == 'up': manual_rc_cmd = [0, 0, speed, 0]
                    elif command == 'down': manual_rc_cmd = [0, 0, -speed, 0]
                    elif command == 'cw': manual_rc_cmd = [0, 0, 0, speed]
                    elif command == 'ccw': manual_rc_cmd = [0, 0, 0, -speed]
                    elif command == 'x': manual_rc_cmd = [0, 0, 0, 0] # 停止

            # --- 制御・推論ロジック ---
            if is_flying:
                start_time = time.perf_counter()
                target_data = detector.detect(frame)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000.0
                log_data["inference_latency_ms"] = f"{latency_ms:.2f}"
                
                # 描画とデータ抽出
                if target_data:
                    bbox = target_data['bbox']
                    keypoints = target_data['keypoints']
                    
                    # 動画への描画
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                    if len(keypoints) > target_kp_idx:
                        kpx, kpy, kpconf = keypoints[target_kp_idx]
                        if kpconf > 0.4:
                            cv2.circle(frame, (kpx, kpy), 8, (0, 0, 255), -1)

                # コマンドの決定 (自動 or 手動)
                if is_tracking:
                    # 【自律追跡モード】
                    if target_data and len(keypoints) > target_kp_idx and keypoints[target_kp_idx][2] > 0.4:
                        kpx, kpy, kpconf = keypoints[target_kp_idx]
                        
                        tracker_output = tracker.update(kpx, kpy, target_data['area'], frame.shape[1], frame.shape[0])
                        rc_cmd = tracker_output[:4]
                        
                        log_data.update({
                            "target_found": True, "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_w": bbox[2], "bbox_h": bbox[3], "confidence": bbox[4],
                            "error_yaw": f"{tracker_output[4]:.2f}", "error_pitch": f"{tracker_output[5]:.2f}", "error_throttle": f"{tracker_output[6]:.2f}"
                        })
                    else:
                        rc_cmd = [0, 0, 0, 0]
                else:
                    # 【手動操作モード】
                    rc_cmd = manual_rc_cmd

                # 機体へコマンド送信
                tello.send_rc_control(int(rc_cmd[0]), int(rc_cmd[1]), int(rc_cmd[2]), int(rc_cmd[3]))
                log_data.update({"rc_lr": rc_cmd[0], "rc_fb": rc_cmd[1], "rc_ud": rc_cmd[2], "rc_y": rc_cmd[3]})

            logger.write_frame(frame)
            loop_duration_ms = (time.perf_counter() - loop_start_time) * 1000.0
            log_data["loop_duration_ms"] = f"{loop_duration_ms:.2f}"
            logger.log_data(log_data)
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[Debug] UNEXPECTED ERROR: {e}")
    finally:
        stop_event.set()
        if input_thread and input_thread.is_alive():
            input_thread.join(timeout=1)
            
        if tello:
            try:
                if 'is_flying' in locals() and is_flying:
                    tello.land()
            except Exception as e:
                print(f"[Debug] Ignored land error: {e}")
            
            try:
                tello.streamoff()
            except Exception:
                pass

            tello.is_flying = False 
            
            try:
                tello.end()
            except Exception:
                pass
            
        if logger:
            logger.close()
            
        if 'detector' in locals() and detector:
            detector.close()

        print("\n[Debug] Program Exited")

if __name__ == "__main__":
    main()