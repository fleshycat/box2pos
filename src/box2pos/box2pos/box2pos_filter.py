# ros2 run your_pkg yolo_ground_localizer.py  (예시)
import math
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, PoseArray, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import Monitoring
from collections import deque
import numpy as np

from yolov8_msgs.msg import Yolov8Inference  # pip/ros msg as in your env

def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=float)

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=float)

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=float)

class MovingAverage2D:
    def __init__(self, window=11):
        self.N = window
        self.bufx = deque(maxlen=window)
        self.bufy = deque(maxlen=window)

    def push(self, x, y):
        self.bufx.append(float(x)); self.bufy.append(float(y))
        # 평균 (창이 덜 찼을 때도 현재까지 평균)
        return (float(np.mean(self.bufx)), float(np.mean(self.bufy)))
    
class YoloGroundLocalizer(Node):
    def __init__(self):
        super().__init__('yolo_ground_localizer')

        # --- Camera intrinsics (params you provided)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fx', 655.68563818)
        self.declare_parameter('fy', 656.8373932213)
        self.declare_parameter('cx', 630.0668311460)
        self.declare_parameter('cy', 366.3269405158)

        self.declare_parameter('cam_pitch_deg', -30.0)  # downward tilt
        self.declare_parameter('cam_roll_deg', 0.0)     # given 0
        # yaw follows drone → relative yaw 0

        # --- Topics
        self.declare_parameter('yolo_topic', '/Yolov8_Inference_1')  # 확실하지 않음: 네 실제 토픽으로
        self.declare_parameter('monitor_topic', '/drone1/fmu/out/monitoring')

        # --- Monitoring fields (string names to getattr)
        #    Example from your snippet:
        self.declare_parameter('field_pos_x', 'pos_x')
        self.declare_parameter('field_pos_y', 'pos_y')
        self.declare_parameter('field_pos_z', 'pos_z')
        self.declare_parameter('field_roll', 'roll')
        self.declare_parameter('field_pitch', 'pitch')
        self.declare_parameter('field_yaw', 'head')  # radians

        # --- RViz frame id
        self.declare_parameter('frame_id', 'map')  # 확실하지 않음: 'ned' or 'map'
        
        self.declare_parameter('roi_enabled', True)
        
        self.declare_parameter('roi_min_u', 100)
        self.declare_parameter('roi_max_u', 1180)
        self.declare_parameter('roi_min_v', 200)
        self.declare_parameter('roi_max_v', 700)

        self.roi_enabled = self.get_parameter('roi_enabled').value
        self.roi_min_u = float(self.get_parameter('roi_min_u').value)
        self.roi_max_u = float(self.get_parameter('roi_max_u').value)
        self.roi_min_v = float(self.get_parameter('roi_min_v').value)
        self.roi_max_v = float(self.get_parameter('roi_max_v').value)
        
        # Params to vars
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value

        self.cam_pitch = math.radians(self.get_parameter('cam_pitch_deg').value)
        self.cam_roll = math.radians(self.get_parameter('cam_roll_deg').value)

        self.yolo_topic = self.get_parameter('yolo_topic').value
        self.monitor_topic = self.get_parameter('monitor_topic').value

        self.field_pos_x = self.get_parameter('field_pos_x').value
        self.field_pos_y = self.get_parameter('field_pos_y').value
        self.field_pos_z = self.get_parameter('field_pos_z').value
        self.field_roll = self.get_parameter('field_roll').value
        self.field_pitch = self.get_parameter('field_pitch').value
        self.field_yaw = self.get_parameter('field_yaw').value

        self.frame_id = self.get_parameter('frame_id').value
        
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

        # State from monitor
        self.C_ned = np.zeros(3, dtype=float)   # [x,y,z]
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Filters
        self.ma = MovingAverage2D(window=11)
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, 'detections_markers', 10)
        self.pose_pub = self.create_publisher(PoseArray, 'detections_posearray', 10)
        self.point_pub = self.create_publisher(PointStamped, 'detection_point', 10)

        # Subscribers
        self.create_subscription(Yolov8Inference, self.yolo_topic, self.on_yolo, reliable_qos)

        # NOTE: 실제 모니터링 메시지 타입을 모름(확실하지 않음). 아래는 예시용 lamdba로 교체.
        # 네 환경의 msg type으로 바꿔서 같은 콜백 로직을 사용해줘.
        qos = QoSProfile(depth=10)

        # ---- 예시: Generic subscription via topic_tools/relay가 아니라면, 실제 타입을 넣어야 함
        self.create_subscription(Monitoring, self.monitor_topic, self.on_monitor, best_effort)

        # 임시 타이머 경고 (실제 타입 연결 필요)
        self.warned_monitor = False
        self.create_timer(2.0, self._warn_monitor_once)

        # Precompute camera->body alignment (no tilt): b_x<-c_z, b_y<-c_x, b_z<-c_y
        self.R_b_c_align = np.array([[0,0,1],
                                     [1,0,0],
                                     [0,1,0]], dtype=float)

        # Fixed gimbal pitch/roll relative to body
        self.R_b_cam_fixed = rot_y(self.cam_pitch) @ rot_x(self.cam_roll) @ self.R_b_c_align

        self.get_logger().info('yolo_ground_localizer ready.')

    def _warn_monitor_once(self):
        if not self.warned_monitor:
            self.get_logger().warn('Monitoring topic subscriber not set (message type unknown). '
                                   'Replace on_monitor subscription with your real msg type.', throttle_duration_sec=0)
            self.warned_monitor = True

    # ---- Replace this with a real callback once you know the message type
    def on_monitor(self, msg):
        # Example mapping based on your snippet:
        self.C_ned[0] = float(getattr(msg, self.field_pos_x))
        self.C_ned[1] = float(getattr(msg, self.field_pos_y))
        self.C_ned[2] = float(getattr(msg, self.field_pos_z))
        self.roll      = float(getattr(msg, self.field_roll))
        self.pitch     = float(getattr(msg, self.field_pitch))
        self.yaw       = float(getattr(msg, self.field_yaw))

    def in_roi(self, u, v):
        if not self.roi_enabled:
            return True
        return (self.roi_min_u <= u <= self.roi_max_u) and (self.roi_min_v <= v <= self.roi_max_v)

    def on_yolo(self, msg: Yolov8Inference):
        # Build body->NED rotation (Z-Y-X)
        # R_n_b = rot_z(self.yaw) @ rot_y(self.pitch) @ rot_x(self.roll)

        # PoseArray for all detections
        poses = PoseArray()
        poses.header = Header()
        poses.header.frame_id = self.frame_id
        poses.header.stamp = self.get_clock().now().to_msg()

        # Markers
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        m_id = 0

        # Drone point marker (optional)
        drone_marker = Marker()
        drone_marker.header.frame_id = self.frame_id
        drone_marker.header.stamp = now
        drone_marker.ns = "drone"
        drone_marker.id = m_id; m_id += 1
        drone_marker.type = Marker.SPHERE
        drone_marker.pose.position.x = float(self.C_ned[0])
        drone_marker.pose.position.y = float(self.C_ned[1])
        drone_marker.pose.position.z = -float(self.C_ned[2])
        drone_marker.scale.x = drone_marker.scale.y = drone_marker.scale.z = 0.2
        drone_marker.color.r = 0.2; drone_marker.color.g = 0.4; drone_marker.color.b = 1.0; drone_marker.color.a = 0.9
        markers.markers.append(drone_marker)

        # Process each detection
        for det in msg.yolov8_inference:
            u = (det.left + det.right) * 0.5
            # v = (det.top  + det.bottom) * 0.5
            v = float(det.bottom)
            
            if not self.in_roi(u, v):
                continue
    
            # Ray in camera frame (pinhole; no distortion)
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
            r_cam = np.array([x, y, 1.0], dtype=float)
            r_cam /= np.linalg.norm(r_cam)

            # Camera->Body (alignment + fixed gimbal pitch/roll)
            r_body = self.R_b_cam_fixed @ r_cam

            # Body->NED
            # r_ned = R_n_b @ r_body
            
            R_n_yaw = rot_z(self.yaw)
            R_cam_fixed = rot_y(self.cam_pitch) @ self.R_b_c_align
            r_ned = R_n_yaw @ (R_cam_fixed @ r_cam)

            # Intersect with ground plane z=0
            Cz = float(self.C_ned[2])
            rz = float(r_ned[2])

            if abs(rz) < 1e-6:
                self.get_logger().warn('Ray nearly parallel to ground; skipping this detection.')
                continue

            t = (0.0 - Cz) / rz

            if t <= 0.0:
                # If this triggers consistently, your z sign convention likely differs.
                self.get_logger().warn('Intersection behind camera (t<=0). '
                                       'Check NED z sign (is z down positive?). Skipping.')
                continue

            P = self.C_ned + t * r_ned  # [x,y,z], z≈0
            Px_raw, Py_raw = float(P[0]), float(P[1])

            Px_avg_tmp, Py_avg_tmp = self.ma.push(Px_raw, Py_raw)
            if abs(Px_raw - Px_avg_tmp) > 3.0: Px_raw = Px_avg_tmp
            if abs(Py_raw - Py_avg_tmp) > 3.0: Py_raw = Py_avg_tmp

            # 최종 이동평균
            Px_ma, Py_ma = self.ma.push(Px_raw, Py_raw)
            P[0] = Px_ma
            P[1] = Py_ma

            # Pose for PoseArray (orientation set identity)
            pose = Pose()
            pose.position.x = float(P[0])
            pose.position.y = float(P[1])
            pose.position.z = float(P[2])
            poses.poses.append(pose)

            # PointStamped (first one only, optional)
            pt = PointStamped()
            pt.header.frame_id = self.frame_id
            pt.header.stamp = now
            pt.point = Point(x=float(P[0]), y=float(P[1]), z=float(P[2]))
            self.point_pub.publish(pt)

            # Marker: point
            mk = Marker()
            mk.header.frame_id = self.frame_id
            mk.header.stamp = now
            mk.ns = "detections"
            mk.id = m_id; m_id += 1
            mk.type = Marker.SPHERE
            mk.pose.position.x = float(P[0])
            mk.pose.position.y = float(P[1])
            mk.pose.position.z = float(P[2])
            mk.scale.x = mk.scale.y = mk.scale.z = 0.25
            mk.color.r = 1.0; mk.color.g = 0.2; mk.color.b = 0.2; mk.color.a = 0.95
            markers.markers.append(mk)

            # Marker: label
            txt = Marker()
            txt.header.frame_id = self.frame_id
            txt.header.stamp = now
            txt.ns = "labels"
            txt.id = m_id; m_id += 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.text = getattr(det, 'class_name', 'obj')
            txt.pose.position.x = float(P[0])
            txt.pose.position.y = float(P[1])
            txt.pose.position.z = float(P[2]) + 0.5
            txt.scale.z = 0.4
            txt.color.r = 1.0; txt.color.g = 1.0; txt.color.b = 1.0; txt.color.a = 0.9
            markers.markers.append(txt)

            # Marker: line from drone to object
            ln = Marker()
            ln.header.frame_id = self.frame_id
            ln.header.stamp = now
            ln.ns = "rays"
            ln.id = m_id; m_id += 1
            ln.type = Marker.LINE_LIST
            ln.scale.x = 0.03
            ln.color.r = 0.3; ln.color.g = 1.0; ln.color.b = 0.3; ln.color.a = 0.9
            ln.points = [Point(x=float(self.C_ned[0]), y=float(self.C_ned[1]), z=-float(self.C_ned[2])),
                         Point(x=float(P[0]), y=float(P[1]), z=float(P[2]))]
            markers.markers.append(ln)

        # Publish
        self.pose_pub.publish(poses)
        self.marker_pub.publish(markers)

def main():
    rclpy.init()
    node = YoloGroundLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
