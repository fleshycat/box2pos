#!/usr/bin/env python3

# YOLOv8 ROS2 publisher (person-only, via classes filter) - GPU ONLY

import os
import torch
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolov8_msgs.msg import InferenceResult, Yolov8Inference
from px4_msgs.srv import ModeChange

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2

bridge = CvBridge()

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        # ---------- GPU ONLY ENFORCEMENT ----------
        # CUDA가 없으면 바로 종료 (CPU 사용 금지)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA가 감지되지 않았습니다. 이 노드는 GPU 전용이며 CPU 실행을 허용하지 않습니다."
            )

        # (선택) 특정 GPU만 사용하고 싶으면 환경변수로 고정 가능
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.device = 0  # CUDA:0
        self.use_half = True # FP32
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        # ---------------- Params ----------------
        self.declare_parameter('system_id', 1)
        self.declare_parameter('image_topic', '/recon_1/camera/image_raw')
        self.declare_parameter('model_path', '/home/user/Projects/tmp_ros2_ws/box2pos_ws/src/yolobot_recognition/scripts/yolov8n.pt')
        self.declare_parameter('conf', 0.4)  # confidence threshold

        self.system_id   = self.get_parameter('system_id').get_parameter_value().integer_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path       = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf        = float(self.get_parameter('conf').get_parameter_value().double_value)

        # ---------------- Model (GPU로 로드) ----------------
        self.model = YOLO(model_path)
        # 모델을 미리 GPU로 옮기고(fuse는 가능 시) FP16 사용
        try:
            self.model.to('cuda:0')
        except Exception as e:
            raise RuntimeError(f"모델을 GPU로 이동하지 못했습니다: {e}")
        try:
            self.model.fuse()
        except Exception:
            pass

        # person 클래스 ID 안전 획득
        names = self.model.names  # dict or list
        if isinstance(names, dict):
            self.person_cls_id = [k for k, v in names.items() if v == 'person'][0]
        else:
            self.person_cls_id = [i for i, v in enumerate(names) if v == 'person'][0]
        self.get_logger().info(
            f'GPU ONLY | class filter: person -> id {self.person_cls_id}, conf={self.conf}, half={self.use_half}'
        )

        # ---------------- Pub/Sub ----------------

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.yolov8_inference = Yolov8Inference()

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.camera_callback,
            qos_profile,
        )

        self.yolov8_pub = self.create_publisher(
            Yolov8Inference,
            f"/Yolov8_Inference_{self.system_id}",
            1
        )
        self.img_pub = self.create_publisher(
            Image,
            f"/inference_result_{self.system_id}",
            qos_profile,
        )

        # (선택) Drone manager service client (기존 유지)
        self.topic_prefix_manager_ = f"drone{self.system_id}/manager/"
        self.mode_change_client = self.create_client(ModeChange, f'{self.topic_prefix_manager_}srv/mode_change')

        

    def send_request(self, mode: int):
        req = ModeChange.Request()
        req.suv_mode = mode
        return self.mode_change_client.call_async(req)

    # --------------- Callback ----------------
    def camera_callback(self, data: Image):
        img = bridge.imgmsg_to_cv2(data, "bgr8")

        # GPU ONLY 추론 (device=0, half=True 강제)
        results = self.model(
            img,
            classes=[self.person_cls_id],
            conf=self.conf,
            device=self.device,
            half=self.use_half,
            verbose=False
        )

        # 헤더 채우기
        self.yolov8_inference.header.frame_id = "inference"
        self.yolov8_inference.header.stamp = self.get_clock().now().to_msg()

        # 결과 순회
        for r in results:
            for box in r.boxes:
                # box.cls는 텐서일 수 있음 → 안전 변환
                cls_tensor = box.cls
                c = int(cls_tensor.item() if hasattr(cls_tensor, 'item') else cls_tensor)
                cls_name = self.model.names[c]
                if cls_name != "person":
                    continue

                # xyxy: (x1, y1, x2, y2) = (left, top, right, bottom)
                # 텐서를 CPU로 안전 변환
                xyxy = box.xyxy
                if hasattr(xyxy, 'detach'):
                    x1, y1, x2, y2 = xyxy[0].detach().to('cpu').numpy().astype(int)
                else:
                    x1, y1, x2, y2 = map(int, xyxy)

                inf = InferenceResult()
                inf.class_name = "person"
                inf.left   = int(x1)
                inf.top    = int(y1)
                inf.right  = int(x2)
                inf.bottom = int(y2)
                # (선택)
                self.yolov8_inference.yolov8_inference.append(inf)

                # (선택) 특정 조건 시 매니저 호출
                # if x1 >= 50:
                #     self.send_request(2)

        # 어노테이션 이미지( Ultralytics helper 사용 )
        annotated_frame = results[0].plot() if len(results) > 0 else img
        img_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.img_pub.publish(img_msg)
    
        # annotated = img.copy()
        # color = (0, 255, 0)
        # for inf in self.yolov8_inference.yolov8_inference:
        #     cv2.rectangle(annotated, (inf.left, inf.top), (inf.right, inf.bottom), color, 2)
        # img_msg = bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        # self.img_pub.publish(img_msg)

        # bbox 메시지 publish
        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

def main():
    rclpy.init(args=None)
    node = CameraSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
