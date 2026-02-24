import cv2
import numpy as np

try:
    from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("[Warning] hailo_platform not found. Hailo mode will strictly fail.")


class PoseEstPostProcessing:
    """
    提供された pose_estimation_utils.py を基にした
    YOLOv8-Pose モデル出力のデコード・後処理クラス
    """
    def __init__(self, max_detections: int = 100, score_threshold: float = 0.5, nms_iou_thresh: float = 0.45,
                 regression_length: int = 15, strides: list = [8, 16, 32]):
        self.max_detections = max_detections
        self.score_threshold = score_threshold
        self.nms_iou_thresh = nms_iou_thresh
        self.regression_length = regression_length
        self.strides = strides

    def post_process(self, raw_detections: dict, height: int, width: int, class_num: int = 1) -> dict:
        raw_detections_keys = list(raw_detections.keys())
        # テンソルのShapeから対応するレイヤーを特定する
        layer_from_shape = {raw_detections[key].shape: key for key in raw_detections_keys}
        detection_output_channels = (self.regression_length + 1) * 4
        keypoints = 51 # 17 keypoints * 3 (x, y, conf)
        
        # 順番: [bbox(stride=32), class(stride=32), kpts(stride=32), bbox(16), class(16), kpts(16), bbox(8), class(8), kpts(8)]
        endnodes = [
            raw_detections[layer_from_shape[(1, 20, 20, detection_output_channels)]],
            raw_detections[layer_from_shape[(1, 20, 20, class_num)]],
            raw_detections[layer_from_shape[(1, 20, 20, keypoints)]],
            raw_detections[layer_from_shape[(1, 40, 40, detection_output_channels)]],
            raw_detections[layer_from_shape[(1, 40, 40, class_num)]],
            raw_detections[layer_from_shape[(1, 40, 40, keypoints)]],
            raw_detections[layer_from_shape[(1, 80, 80, detection_output_channels)]],
            raw_detections[layer_from_shape[(1, 80, 80, class_num)]],
            raw_detections[layer_from_shape[(1, 80, 80, keypoints)]]
        ]

        return self.extract_pose_estimation_results(endnodes, height, width, class_num)

    def extract_pose_estimation_results(self, endnodes: list, height: int, width: int, class_num: int) -> dict:
        batch_size = endnodes[0].shape[0]
        strides = self.strides[::-1]
        image_dims = (height, width)

        raw_boxes = endnodes[:7:3]
        scores = [np.reshape(s, (-1, s.shape[1] * s.shape[2], class_num)) for s in endnodes[1:8:3]]
        scores = np.concatenate(scores, axis=1)

        kpts = [np.reshape(c, (-1, c.shape[1] * c.shape[2], 17, 3)) for c in endnodes[2:9:3]]

        decoded_boxes, decoded_kpts = self.decoder(raw_boxes, kpts, strides, image_dims, self.regression_length)
        decoded_kpts = np.reshape(decoded_kpts, (batch_size, -1, 51))
        predictions = np.concatenate([decoded_boxes, scores, decoded_kpts], axis=2)

        nms_res = self.non_max_suppression(
            predictions, conf_thres=self.score_threshold,
            iou_thres=self.nms_iou_thresh, max_det=self.max_detections
        )

        output = {
            'bboxes': np.zeros((batch_size, self.max_detections, 4)),
            'keypoints': np.zeros((batch_size, self.max_detections, 17, 2)),
            'joint_scores': np.zeros((batch_size, self.max_detections, 17, 1)),
            'scores': np.zeros((batch_size, self.max_detections, 1)),
            'num_detections': np.zeros((batch_size,), dtype=int)
        }

        for b in range(batch_size):
            num_det = nms_res[b]['num_detections']
            output['num_detections'][b] = num_det
            if num_det > 0:
                output['bboxes'][b, :num_det] = nms_res[b]['bboxes']
                output['keypoints'][b, :num_det] = nms_res[b]['keypoints'][..., :2]
                output['joint_scores'][b, :num_det, ..., 0] = self._sigmoid(nms_res[b]['keypoints'][..., 2])
                output['scores'][b, :num_det, ..., 0] = nms_res[b]['scores']

        return output

    def decoder(self, raw_boxes, raw_kpts, strides, image_dims, reg_max):
        boxes = None
        decoded_kpts = None

        for box_distribute, kpts, stride, _ in zip(raw_boxes, raw_kpts, strides, np.arange(3)):
            shape = [int(x / stride) for x in image_dims]
            grid_x = np.arange(shape[1]) + 0.5
            grid_y = np.arange(shape[0]) + 0.5
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            ct_row = grid_y.flatten() * stride
            ct_col = grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            reg_range = np.arange(reg_max + 1)
            box_distribute = np.reshape(box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1))
            box_distance = self._softmax(box_distribute) * np.reshape(reg_range, (1, 1, 1, -1))
            box_distance = np.sum(box_distance, axis=-1) * stride

            box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
            decode_box = np.expand_dims(center, axis=0) + box_distance

            xmin, ymin, xmax, ymax = decode_box[:, :, 0], decode_box[:, :, 1], decode_box[:, :, 2], decode_box[:, :, 3]
            
            xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
            boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)

            kpts[..., :2] *= 2
            kpts[..., :2] = stride * (kpts[..., :2] - 0.5) + np.expand_dims(center[..., :2], axis=1)
            decoded_kpts = kpts if decoded_kpts is None else np.concatenate([decoded_kpts, kpts], axis=1)

        return boxes, decoded_kpts

    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def _softmax(self, x): return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)
    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def nms(self, dets, thresh):
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]

        suppressed = np.zeros(dets.shape[0], dtype=int)
        for i in range(len(order)):
            idx_i = order[i]
            if suppressed[idx_i] == 1: continue
            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if suppressed[idx_j] == 1: continue

                xx1 = max(x1[idx_i], x1[idx_j])
                yy1 = max(y1[idx_i], y1[idx_j])
                xx2 = min(x2[idx_i], x2[idx_j])
                yy2 = min(y2[idx_i], y2[idx_j])
                w = max(0.0, xx2 - xx1 + 1)
                h = max(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[idx_i] + areas[idx_j] - inter)

                if ovr >= thresh:
                    suppressed[idx_j] = 1

        return np.where(suppressed == 0)[0]

    def non_max_suppression(self, prediction, conf_thres=0.1, iou_thres=0.45, max_det=100, n_kpts=17):
        nc = prediction.shape[2] - n_kpts * 3 - 4
        xc = prediction[..., 4] > conf_thres
        ki = 4 + nc
        output = []

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                output.append({'bboxes': np.zeros((0, 4)), 'keypoints': np.zeros((0, n_kpts, 3)), 'scores': np.zeros((0)), 'num_detections': 0})
                continue

            boxes = self.xywh2xyxy(x[:, :4])
            kpts = x[:, ki:]

            conf = np.expand_dims(x[:, 4:ki].max(1), 1)
            j = np.expand_dims(x[:, 4:ki].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, kpts), 1)[keep]
            x = x[x[:, 4].argsort()[::-1][:max_det]]

            if not x.shape[0]:
                output.append({'bboxes': np.zeros((0, 4)), 'keypoints': np.zeros((0, n_kpts, 3)), 'scores': np.zeros((0)), 'num_detections': 0})
                continue

            boxes = x[:, :4]
            scores = x[:, 4]
            kpts = x[:, 6:].reshape(-1, n_kpts, 3)

            i = self.nms(np.concatenate((boxes, np.expand_dims(scores, 1)), axis=1), iou_thres)
            output.append({'bboxes': boxes[i], 'keypoints': kpts[i], 'scores': scores[i], 'num_detections': len(i)})

        return output


class HailoPoseDetector:
    TARGET_CLASS_ID = 0
    INPUT_H = 640
    INPUT_W = 640
    
    def __init__(self, hef_path='models/yolov8m_pose.hef', threshold=0.5):
        if not HAILO_AVAILABLE: raise RuntimeError("[Error] Hailo libs missing")
        
        print(f"[Debug] Loading Hailo Pose HEF: {hef_path}")
        self.threshold = threshold

        self.post_processor = PoseEstPostProcessing(
            max_detections=100, 
            score_threshold=threshold, 
            nms_iou_thresh=0.5,
            regression_length=15, 
            strides=[8, 16, 32]
        )
        
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
        self.target = VDevice(params=params)
        
        self.hef = HEF(hef_path)
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_info = self.hef.get_output_vstream_infos()[0]
        
        config_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, config_params)[0]
        
        input_params = InputVStreamParams.make(self.network_group, quantized=True, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(self.network_group, quantized=True, format_type=FormatType.FLOAT32)
        
        self.pipeline = InferVStreams(self.network_group, input_params, output_params)
        self.network = self.network_group.activate(self.network_group.create_params())
        self.pipeline.__enter__()
        self.network.__enter__()
        print("[Debug] Hailo Pose Detector initialized for 640x640 input.")

    def detect(self, frame):
        h, w = self.INPUT_H, self.INPUT_W
        orig_h, orig_w = frame.shape[:2]
        
        # 前処理
        resized = cv2.resize(frame, (w, h))
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = {self.input_info.name: np.expand_dims(resized_rgb, axis=0)}
        
        # 推論
        raw_results = self.pipeline.infer(input_data)
        
        # デコード
        results = self.post_processor.post_process(raw_results, h, w, class_num=1)
        
        # バッチサイズ1を想定して取り出す
        num_det = results['num_detections'][0]
        if num_det == 0:
            return None
            
        bboxes = results['bboxes'][0][:num_det]
        keypoints = results['keypoints'][0][:num_det]
        joint_scores = results['joint_scores'][0][:num_det]
        scores = results['scores'][0][:num_det]
        
        all_person_detections = []
        
        # 推論用(640x640)の座標から、入力画像元の解像度へスケールを戻す
        scale_x = orig_w / w
        scale_y = orig_h / h

        for box, score, kpts, kpt_scores in zip(bboxes, scores, keypoints, joint_scores):
            if score[0] < self.threshold:
                continue
            
            xmin, ymin, xmax, ymax = box
            
            # 元解像度の座標にマッピング
            xmin = max(0, int(xmin * scale_x))
            xmax = min(orig_w, int(xmax * scale_x))
            ymin = max(0, int(ymin * scale_y))
            ymax = min(orig_h, int(ymax * scale_y))
            
            box_w = xmax - xmin
            box_h = ymax - ymin
            
            mapped_kpts = []
            for kp, kp_score in zip(kpts, kpt_scores):
                kpx = max(0, min(orig_w, int(kp[0] * scale_x)))
                kpy = max(0, min(orig_h, int(kp[1] * scale_y)))
                mapped_kpts.append((kpx, kpy, float(kp_score[0])))
                
            all_person_detections.append({
                'bbox': [xmin, ymin, box_w, box_h, float(score[0])],
                'keypoints': mapped_kpts,
                'area': box_w * box_h
            })
        
        if not all_person_detections:
            return None

        # 画面内で最も面積が大きい人物をターゲットに選定
        new_target = max(all_person_detections, key=lambda d: d['area'])
        
        return new_target

    def close(self):
        if HAILO_AVAILABLE:
            if hasattr(self, 'network'): self.network.__exit__(None, None, None)
            if hasattr(self, 'pipeline'): self.pipeline.__exit__(None, None, None)