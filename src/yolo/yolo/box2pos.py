#!/usr/bin/env python3
import math
from typing import Optional, List
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from yolov8_msgs.msg import Yolov8Inference
from px4_msgs.msg import Monitoring, TrajectorySetpoint

def rpy_deg_to_R_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r = math.radians(roll_deg); p = math.radians(pitch_deg); y = math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

class BBox:
    __slots__ = ('l', 't', 'r', 'b', 'cls')
    def __init__(self, left, top, right, bottom, cls):
        self.l = float(left); self.t = float(top)
        self.r = float(right); self.b = float(bottom)
        self.cls = (cls or "").lower()

class BBoxToGroundNED(Node):
    def __init__(self):
        super().__init__('bbox_to_ground_ned')

        # --- Parameters ---
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('bbox_topic', '/Yolov8_Inference_1')
        self.declare_parameter('monitor_topic', '/drone1/fmu/out/monitoring')
        self.declare_parameter('tag_topic', '/drone1/jfi/in/target')  # TrajectorySetpoint(NED)로 가정
        self.declare_parameter('class_filter', 'person')              # 필요 시 ''로 비활성화

        # 카메라→기체 회전 (기본: pitch 45° 아래로 보는 가정, 실제 짐벌 각도로 교체 권장)
        self.declare_parameter('cam_roll_deg', 0.0)
        self.declare_parameter('cam_pitch_deg', -30.0)
        self.declare_parameter('cam_yaw_deg', 0.0)
        self.declare_parameter('D_ground', 0.0)  # NED Down(+), 지면이 0이면 0

        # Monitoring 메시지에서 읽어올 필드명
        self.declare_parameter('field_pos_x', 'pos_x')
        self.declare_parameter('field_pos_y', 'pos_y')
        self.declare_parameter('field_pos_z', 'pos_z')
        self.declare_parameter('field_ref_lat', 'ref_lat')
        self.declare_parameter('field_ref_lon', 'ref_lon')
        self.declare_parameter('field_ref_alt', 'ref_alt')
        self.declare_parameter('field_roll_deg', 'roll')
        self.declare_parameter('field_pitch_deg', 'pitch')
        self.declare_parameter('field_yaw_deg', 'head')

        # 픽셀→지상점 매칭 성공 판정 임계
        self.declare_parameter('image_threshold', 20.0)

        # --- Camera intrinsics (CameraInfo가 없으므로 상수 주입; 1차 캘리브레이션 기본) ---
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fx', 655.68563818)
        self.declare_parameter('fy', 656.8373932213)
        self.declare_parameter('cx', 630.0668311460)
        self.declare_parameter('cy', 366.3269405158)
        # 왜곡계수(현재 투영에 미사용, 필요시 undistort 경로에서 사용)
        self.declare_parameter('dist_k1', -0.104180068924895)
        self.declare_parameter('dist_k2',  0.103637243022842)
        self.declare_parameter('dist_p1', -0.001048795800793)
        self.declare_parameter('dist_p2', -0.003752605534416)
        self.declare_parameter('dist_k3', -0.037916420163170)

        # --- Load params ---
        self.cam_info_topic = self.get_parameter('camera_info_topic').value
        self.bbox_topic     = self.get_parameter('bbox_topic').value
        self.monitor_topic  = self.get_parameter('monitor_topic').value
        self.tag_topic      = self.get_parameter('tag_topic').value
        self.cls_filter     = self.get_parameter('class_filter').value.strip().lower()

        self.R_bc = rpy_deg_to_R_zyx(
            self.get_parameter('cam_roll_deg').value,
            self.get_parameter('cam_pitch_deg').value,
            self.get_parameter('cam_yaw_deg').value,
        )
        self.D_ground = float(self.get_parameter('D_ground').value)

        self.field_pos_x = self.get_parameter('field_pos_x').value
        self.field_pos_y = self.get_parameter('field_pos_y').value
        self.field_pos_z = self.get_parameter('field_pos_z').value
        self.field_ref_lat = self.get_parameter('field_ref_lat').value
        self.field_ref_lon = self.get_parameter('field_ref_lon').value
        self.field_ref_alt = self.get_parameter('field_ref_alt').value
        self.field_roll  = self.get_parameter('field_roll_deg').value
        self.field_pitch = self.get_parameter('field_pitch_deg').value
        self.field_yaw   = self.get_parameter('field_yaw_deg').value

        self.image_threshold = float(self.get_parameter('image_threshold').value)

        # --- State ---
        self.K: Optional[np.ndarray] = None
        self.W: Optional[int] = None
        self.H: Optional[int] = None
        self.C_refllh = np.zeros(3, dtype=float)  # [lat, lon, alt]
        self.C_ned = np.zeros(3, dtype=float)     # [N,E,D]
        self.T_ned = np.zeros(3, dtype=float)     # [N,E,D]
        self.R_nb  = np.eye(3)
        self.flag = False

        # CameraInfo가 없어도 동작하도록 파라미터에서 즉시 초기화
        self.W = int(self.get_parameter('width').value)
        self.H = int(self.get_parameter('height').value)
        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        self.K = np.array([[fx, 0.0, cx],
                           [0.0, fy, cy],
                           [0.0, 0.0, 1.0]], dtype=float)

        # --- QoS ---
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Subs ---
        # CameraInfo가 나올 경우 K/W/H 덮어쓰기(없어도 무관)
        self.create_subscription(CameraInfo, self.cam_info_topic, self.cb_cam_info, 10)
        # YOLO bbox는 퍼블리셔 기본이 RELIABLE이므로 RELIABLE로 구독
        self.create_subscription(Yolov8Inference, self.bbox_topic, self.cb_bboxes, reliable_qos)
        # 센서/상태류는 BEST_EFFORT
        self.create_subscription(Monitoring, self.monitor_topic, self.cb_monitor, best_effort)
        self.create_subscription(TrajectorySetpoint, self.tag_topic, self.cb_tag, best_effort)
        self.create_subscription(Image, "/inference_result_1", self.cb_image, best_effort)

        # --- Pubs ---
        self.pub_points  = self.create_publisher(PointStamped, '/ground_objects/points', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/ground_objects/markers', 10)
        self.pub_images  = self.create_publisher(Image, '/ground_objects/inference_result_1', 10)

        self.get_logger().info(f'started. bbox_topic={self.bbox_topic}, K=({fx:.3f},{fy:.3f},{cx:.3f},{cy:.3f}), size={self.W}x{self.H}')

    # ---------- Callbacks ----------
    def cb_cam_info(self, msg: CameraInfo):
        pass
        # if len(msg.k) == 9:
        #     self.K = np.array([[msg.k[0], msg.k[1], msg.k[2]],
        #                        [msg.k[3], msg.k[4], msg.k[5]],
        #                        [msg.k[6], msg.k[7], msg.k[8]]], dtype=float)
        # self.W, self.H = int(msg.width), int(msg.height)

    def cb_monitor(self, msg: Monitoring):
        try:
            self.C_ned[0] = float(getattr(msg, self.field_pos_x))
            self.C_ned[1] = float(getattr(msg, self.field_pos_y))
            self.C_ned[2] = float(getattr(msg, self.field_pos_z))
            self.C_refllh[0] = float(getattr(msg, self.field_ref_lat))
            self.C_refllh[1] = float(getattr(msg, self.field_ref_lon))
            self.C_refllh[2] = float(getattr(msg, self.field_ref_alt))
        except Exception:
            return
        roll  = self._safe_get_deg(msg, self.field_roll,  0.0)
        pitch = self._safe_get_deg(msg, self.field_pitch, 0.0)
        yaw   = self._safe_get_deg(msg, self.field_yaw,   0.0)
        self.R_nb = rpy_deg_to_R_zyx(roll, pitch, yaw)

    def cb_tag(self, msg: TrajectorySetpoint):
        # TrajectorySetpoint.position = [N,E,D] (float[3]) 로 가정
        try:
            self.T_ned = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=float)
        except Exception:
            return

    def _safe_get_deg(self, msg, name: str, default: float) -> float:
        if not name:
            return default
        try:
            return float(getattr(msg, name))
        except Exception:
            return default

    def cb_image(self, msg: Image):
        if self.flag:
            self.pub_images.publish(msg)

    def cb_bboxes(self, msg: Yolov8Inference):
        if self.K is None or self.W is None or self.H is None:
            return

        boxes: List[BBox] = [BBox(it.left, it.top, it.right, it.bottom, it.class_name)
                             for it in msg.yolov8_inference]

        if not boxes:
            self.flag = False
            return

        header = self._derive_header(msg.header)

        # clear markers
        ma = MarkerArray()
        clear = Marker()
        clear.header = header; clear.ns = 'ground_objects'
        clear.id = 0; clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        R_nc = self.R_nb @ self.R_bc
        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]

        mid = 1
        any_ok = False
        for b in boxes:
            if self.cls_filter and b.cls != self.cls_filter:
                continue

            # 바닥 교차점은 bbox 하변 중심(v = bottom) 사용
            u = 0.5 * (b.l + b.r)
            v = min(max(b.b, 0.0), float(self.H - 1))

            x = (u - cx) / fx
            y = (v - cy) / fy
            d_c = np.array([x, y, 1.0], dtype=float)
            d_n = R_nc @ d_c
            dz = d_n[2]
            if abs(dz) < 1e-9:
                continue

            # 카메라 위치 C_ned에서 지면 D_ground와의 교점
            t = (self.D_ground - self.C_ned[2]) / dz
            if t <= 0:
                continue

            P = self.C_ned + t * d_n  # [N,E,D_ground]
            ps = PointStamped()
            ps.header = header
            ps.point.x = float(P[0])
            ps.point.y = float(P[1])
            ps.point.z = float(self.D_ground)  # NED Down(+)
            self.pub_points.publish(ps)

            any_ok = True
            m = Marker()
            m.header = header
            m.ns = 'ground_objects'
            m.id = mid; mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.x = ps.point.x
            m.pose.position.y = ps.point.y
            m.pose.position.z = ps.point.z
            m.scale.x = m.scale.y = m.scale.z = 0.5
            m.color.a = 1.0; m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0
            ma.markers.append(m)

        self.pub_markers.publish(ma)
        self.flag = any_ok and self._image_compare_last(ps)  # ps는 마지막 성공 포인트 기준

    def _derive_header(self, src_header: Header) -> Header:
        h = Header()
        try:
            h = src_header
        except Exception:
            pass
        if not h.frame_id:
            h.frame_id = 'ned'
        return h

    def _image_compare_last(self, ps: PointStamped):
        p_ned = np.array([ps.point.x, ps.point.y, ps.point.z])
        if np.all(np.abs(p_ned - self.T_ned) < self.image_threshold):
            self.flag = True
            return True
        else:
            self.flag = False
            return False

# ---- (옵션) LLH→NED 변환이 필요할 때 사용: 현재 코드는 TrajectorySetpoint(NED) 사용 중이라 미사용 ----
a = 6378137.0
f = 1.0 / 298.257223563
e2 = 2 * f - f * f
def LLH2NED(LLH, ref_LLH):
    lat_ref = np.deg2rad(ref_LLH[0]); lon_ref = np.deg2rad(ref_LLH[1])
    lat = np.deg2rad(LLH[0]); lon = np.deg2rad(LLH[1])
    sin_lat_ref = np.sin(lat_ref); cos_lat_ref = np.cos(lat_ref)
    N_ref = a / np.sqrt(1 - e2 * sin_lat_ref**2)
    dlat = lat - lat_ref; dlon = lon - lon_ref
    NED_N = dlat * N_ref
    NED_E = dlon * N_ref * cos_lat_ref
    NED_D = LLH[2] - ref_LLH[2]
    return np.array([NED_N, NED_E, NED_D])

def main():
    rclpy.init()
    node = BBoxToGroundNED()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


































#### 원본 #####
# #!/usr/bin/env python3

# import math
# from typing import Optional, List
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
# from std_msgs.msg import Header
# from sensor_msgs.msg import CameraInfo, Image
# from geometry_msgs.msg import PointStamped
# from visualization_msgs.msg import Marker, MarkerArray
# from yolov8_msgs.msg import Yolov8Inference  # 고정 import
# from px4_msgs.msg import Monitoring, TrajectorySetpoint  # 모니터링 메시지 타입

# def rpy_deg_to_R_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
#     r = math.radians(roll_deg); p = math.radians(pitch_deg); y = math.radians(yaw_deg)
#     cr, sr = math.cos(r), math.sin(r)
#     cp, sp = math.cos(p), math.sin(p)
#     cy, sy = math.cos(y), math.sin(y)
#     Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
#     Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
#     Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
#     return Rz @ Ry @ Rx
# class BBox:
#     __slots__ = ('l', 't', 'r', 'b', 'cls')
#     def __init__(self, left, top, right, bottom, cls):
#         self.l = float(left); self.t = float(top)
#         self.r = float(right); self.b = float(bottom)
#         self.cls = (cls or "").lower()
# class BBoxToGroundNED(Node):
#     def __init__(self):
#         super().__init__('bbox_to_ground_ned')
#         # Parameters
#         self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
#         self.declare_parameter('bbox_topic', '/Yolov8_Inference_1')  # 필요 시 실행 시점에 변경
#         self.declare_parameter('monitor_topic', '/drone1/fmu/out/monitoring')
#         self.declare_parameter('tag_topic', '/drone1/jfi/in/target')
#         self.declare_parameter('class_filter', 'airplane')
#         self.declare_parameter('cam_roll_deg', 0.0)
#         self.declare_parameter('cam_pitch_deg', 45.0)
#         self.declare_parameter('cam_yaw_deg', 0.0)
#         self.declare_parameter('D_ground', 0.0)

#         self.declare_parameter('field_pos_x', 'pos_x')
#         self.declare_parameter('field_pos_y', 'pos_y')
#         self.declare_parameter('field_pos_z', 'pos_z')

#         self.declare_parameter('field_ref_lat', 'ref_lat')
#         self.declare_parameter('field_ref_lon', 'ref_lon')
#         self.declare_parameter('field_ref_alt', 'ref_alt')

#         self.declare_parameter('field_roll_deg', 'roll')
#         self.declare_parameter('field_pitch_deg', 'pitch')
#         self.declare_parameter('field_yaw_deg', 'head')

#         self.declare_parameter('image_threshold', 20.0)

#         # Load
#         self.cam_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
#         self.bbox_topic     = self.get_parameter('bbox_topic').get_parameter_value().string_value
#         self.monitor_topic  = self.get_parameter('monitor_topic').get_parameter_value().string_value
#         self.tag_topic      = self.get_parameter('tag_topic').get_parameter_value().string_value
#         self.cls_filter     = self.get_parameter('class_filter').get_parameter_value().string_value.strip().lower()
#         self.R_bc = rpy_deg_to_R_zyx(
#             self.get_parameter('cam_roll_deg').get_parameter_value().double_value,
#             self.get_parameter('cam_pitch_deg').get_parameter_value().double_value,
#             self.get_parameter('cam_yaw_deg').get_parameter_value().double_value,
#         )
#         self.D_ground = float(self.get_parameter('D_ground').get_parameter_value().double_value)

#         self.field_pos_x = self.get_parameter('field_pos_x').get_parameter_value().string_value
#         self.field_pos_y = self.get_parameter('field_pos_y').get_parameter_value().string_value
#         self.field_pos_z = self.get_parameter('field_pos_z').get_parameter_value().string_value

#         self.field_ref_lat = self.get_parameter('field_ref_lat').get_parameter_value().string_value
#         self.field_ref_lon = self.get_parameter('field_ref_lon').get_parameter_value().string_value
#         self.field_ref_alt = self.get_parameter('field_ref_alt').get_parameter_value().string_value

#         self.field_roll  = self.get_parameter('field_roll_deg').get_parameter_value().string_value
#         self.field_pitch = self.get_parameter('field_pitch_deg').get_parameter_value().string_value
#         self.field_yaw   = self.get_parameter('field_yaw_deg').get_parameter_value().string_value

#         self.image_threshold = float(self.get_parameter('image_threshold').get_parameter_value().double_value)

#         # State
#         self.K: Optional[np.ndarray] = None
#         self.W: Optional[int] = None
#         self.H: Optional[int] = None
#         self.C_refllh = np.zeros(3, dtype=float)  # [lat, lon, alt]
#         self.C_ned = np.zeros(3, dtype=float)  # [N,E,D]
#         self.T_ned = np.zeros(3, dtype=float)  # [N,E,D]
#         self.R_nb  = np.eye(3)
#         self.flag = False

#         # QoS
#         best_effort = QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=10
#         )
#         # Subs (고정 타입)
#         self.create_subscription(CameraInfo, self.cam_info_topic, self.cb_cam_info, 10)
#         # self.create_subscription(Image, '/camera/camera/color/image_raw', self.)
#         self.create_subscription(Yolov8Inference, self.bbox_topic, self.cb_bboxes, best_effort)
#         self.create_subscription(Monitoring, self.monitor_topic, self.cb_monitor, best_effort)
#         self.create_subscription(TrajectorySetpoint, self.tag_topic, self.cb_tag, best_effort)
#         self.create_subscription(Image, "/inference_result_1", self.cb_image, best_effort)
#         # Pubs
#         self.pub_points  = self.create_publisher(PointStamped, '/ground_objects/points', 10)
#         self.pub_markers = self.create_publisher(MarkerArray, '/ground_objects/markers', 10)
#         self.pub_images = self.create_publisher(Image, '/ground_objects/inference_result_1', 10)
#         self.get_logger().info(f'started. bbox_topic={self.bbox_topic}')
#     # Callbacks
#     def cb_cam_info(self, msg: CameraInfo):
#         if len(msg.k) == 9:
#             self.K = np.array([[msg.k[0], msg.k[1], msg.k[2]],
#                                [msg.k[3], msg.k[4], msg.k[5]],
#                                [msg.k[6], msg.k[7], msg.k[8]]], dtype=float)
#         self.W, self.H = int(msg.width), int(msg.height)
#     def cb_monitor(self, msg):
#         try:
#             self.C_ned[0] = float(getattr(msg, self.field_pos_x))
#             self.C_ned[1] = float(getattr(msg, self.field_pos_y))
#             self.C_ned[2] = float(getattr(msg, self.field_pos_z))
#             self.C_refllh[0] = float(getattr(msg, self.field_ref_lat))
#             self.C_refllh[1] = float(getattr(msg, self.field_ref_lon))
#             self.C_refllh[2] = float(getattr(msg, self.field_ref_alt))

#         except Exception:
#             return
#         roll  = self._safe_get_deg(msg, self.field_roll,  0.0)
#         pitch = self._safe_get_deg(msg, self.field_pitch, 0.0)
#         yaw   = self._safe_get_deg(msg, self.field_yaw,   0.0)
#         self.R_nb = rpy_deg_to_R_zyx(roll, pitch, yaw)
#     def cb_tag(self, msg):
#         try:
#             T_llh = float(getattr(msg, "position"))
#             self.T_ned = LLH2NED(T_llh, self.C_refllh)
#         except Exception:
#             return
#     def _safe_get_deg(self, msg, name: str, default: float) -> float:
#         if not name: return default
#         try: return float(getattr(msg, name))
#         except Exception: return default
#     def cb_image(self, msg: Image):
#         if(self.flag):
#             self.get_logger().info(f"Image Publishing...")
#             self.pub_images.publish(msg)
#             return
#         else:
#             return
#     def cb_bboxes(self, msg: Yolov8Inference):
#         if self.K is None or self.W is None or self.H is None:
#             return

#         # convert to BBox list
#         self.get_logger().info(f"Received bboxes")
#         boxes: List[BBox] = [
#             BBox(it.left, it.top, it.right, it.bottom, it.class_name)
#             for it in msg.yolov8_inference
#         ]

#         if not boxes:
#             self.flag = False
#             return

#         header = self._derive_header(msg.header)
#         # clear markers
#         ma = MarkerArray()
#         clear = Marker()
#         clear.header = header; clear.ns = 'ground_objects'
#         clear.id = 0; clear.action = Marker.DELETEALL
#         ma.markers.append(clear)
#         R_nc = self.R_nb @ self.R_bc
#         fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
#         mid = 1
#         for b in boxes:
#             self.get_logger().info(f"Processing bbox: {b.l}, {b.t}, {b.r}, {b.b}, {b.cls}")
#             # if self.cls_filter and b.cls != self.cls_filter:
#             #     continue
#             u = 0.5 * (b.l + b.r)
#             v = min(max(b.b, 0.0), float(self.H - 1))
#             x = (u - cx) / fx
#             y = (v - cy) / fy
#             d_c = np.array([x, y, 1.0], dtype=float)
#             d_n = R_nc @ d_c
#             dz = d_n[2]
#             self.get_logger().info(f"Computed d_n: {d_n}, dz: {dz}")

#             if abs(dz) < 1e-9:
#                 continue

#             t = (self.D_ground - self.C_ned[2]) / dz
#             self.get_logger().info(f"Computed t: {t}")

#             if t <= 0:
#                 continue

#             P = self.C_ned + t * d_n  # [N,E,D_ground]
#             ps = PointStamped()
#             ps.header = header
#             ps.point.x = float(P[0])
#             ps.point.y = float(P[1])
#             ps.point.z = float(self.D_ground)  # NED Down(+)
#             self.pub_points.publish(ps)
#             self.get_logger().info(
#                 f"PointStamped position: x={ps.point.x:.3f}, y={ps.point.y:.3f}, z={ps.point.z:.3f}"
#             )
#             if self._image_compare(ps):
#                 self.get_logger().info(f"Image Compare Success")

#             m = Marker()
#             m.header = header
#             m.ns = 'ground_objects'
#             m.id = mid; mid += 1
#             m.type = Marker.SPHERE
#             m.action = Marker.ADD
#             m.pose.orientation.w = 1.0
#             m.pose.position.x = ps.point.x
#             m.pose.position.y = ps.point.y
#             m.pose.position.z = ps.point.z
#             m.scale.x = m.scale.y = m.scale.z = 0.5
#             m.color.a = 1.0; m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0
#             ma.markers.append(m)

#         self.pub_markers.publish(ma)

#     def _derive_header(self, src_header: Header) -> Header:
#         h = Header()
#         try:
#             h = src_header
#         except Exception:
#             pass
#         if not h.frame_id:
#             h.frame_id = 'ned'
#         return h

#     def _image_compare(self, ps: PointStamped):
#         p_ned = np.array([ps.point.x, ps.point.y, ps.point.z])
#         if np.all(np.abs(p_ned - self.T_ned) < self.image_threshold):
#             self.get_logger().info(f"Image Compare: True (diff = {np.abs(p_ned - self.T_ned)})")
#             self.flag = True
#             return True
#         else:
#             self.get_logger().warn(f"Image Compare: False (diff = {np.abs(p_ned - self.T_ned)})")
#             self.flag = False
#             return False

# # WGS-84
# a = 6378137.0
# f = 1.0 / 298.257223563
# e2 = 2 * f - f * f

# def LLH2NED(LLH, ref_LLH):
#   lat_ref = np.deg2rad(ref_LLH[0])
#   lon_ref = np.deg2rad(ref_LLH[1])
#   lat = np.deg2rad(LLH[0])
#   lon = np.deg2rad(LLH[1])

#   sin_lat_ref = np.sin(lat_ref)
#   cos_lat_ref = np.cos(lat_ref)

#   N_ref = a / np.sqrt(1 - e2 * sin_lat_ref**2)

#   dlat = lat - lat_ref
#   dlon = lon - lon_ref

#   NED_N = dlat * N_ref
#   NED_E = dlon * N_ref * cos_lat_ref
#   NED_D = LLH[2] - ref_LLH[2]

#   return np.array([NED_N, NED_E, NED_D])

# def main():
#     rclpy.init()
#     node = BBoxToGroundNED()
#     try:
#         rclpy.spin(node)
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
# if __name__ == '__main__':
#     main()
