import time
import numpy as np

class PIDTracker:
    def __init__(self):
        self.pid_params = {
            'yaw':      [0.15, 0.001, 0.035],
            'pitch':    [0.3, 0.002, 0.01],
            'throttle': [0.2, 0.005, 0.06]
        }
        """
        今回の実験では，target_areaを次の順番で大きくしてみる
        1. 540 * 180
        2. 400 * 400
        3. 600 * 600
        4. 800 * 600
        """
        self.target_area = 540 * 180
        self.prev_error = {'yaw': 0, 'pitch': 0, 'throttle': 0}
        self.integral = {'yaw': 0, 'pitch': 0, 'throttle': 0}
        self.last_time = time.time()

    def update(self, center_x, center_y, area, frame_w, frame_h):
        now = time.time()
        dt = now - self.last_time
        if dt == 0: dt = 1e-6
        self.last_time = now

        frame_center_x, frame_center_y = frame_w // 2, frame_h // 2
        errors = {
            'yaw': center_x - frame_center_x,
            'throttle': frame_center_y - center_y,
            'pitch': self.target_area - area
        }

        speeds = {}
        for axis in ['yaw', 'pitch', 'throttle']:
            kp, ki, kd = self.pid_params[axis]
            self.integral[axis] = np.clip(self.integral[axis] + errors[axis] * dt, -200, 200)
            derivative = (errors[axis] - self.prev_error[axis]) / dt
            output = kp * errors[axis] + ki * self.integral[axis] + kd * derivative
            
            limit = 100 if axis == 'yaw' else (50 if axis == 'throttle' else 50)
            speeds[axis] = int(np.clip(output, -limit, limit))
            self.prev_error[axis] = errors[axis]

        return 0, speeds['pitch'], speeds['throttle'], speeds['yaw'], errors['yaw'], errors['pitch'], errors['throttle']#