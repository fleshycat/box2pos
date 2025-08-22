#!/usr/bin/env python3
import math
from typing import Optional, List
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped, Point  # <-- NEW: Point for marker points
from visualization_msgs.msg import Marker, MarkerArray
from yolov8_msgs.msg import Yolov8Inference
from px4_msgs.msg import Monitoring

from scipy.spatial.transform import Rotation as R

# --- Rotation helpers (radians) ---
def Rx(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def Ry(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def Rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]], dtype=float)

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
        self.declare_parameter('bbox_topic', '/Yolov8_Inference_1')
        self.declare_parameter('monitor_topic', '/drone1/fmu/out/monitoring')
        self.declare_parameter('class_filter', 'person')   # '' to disable

        # Camera fixed angles (roll/pitch fixed, yaw aligned to heading=0)
        self.declare_parameter('cam_roll_deg', 0.0)
        self.declare_parameter('cam_pitch_deg', -30.0)

        # Ground plane D (NED Down, usually 0)
        self.declare_parameter('D_ground', 0.0)

        # Monitoring field names (radians for heading)
        self.declare_parameter('field_pos_x', 'pos_x')
        self.declare_parameter('field_pos_y', 'pos_y')
        self.declare_parameter('field_pos_z', 'pos_z')
        self.declare_parameter('field_heading_rad', 'head')

        # Camera intrinsics (fixed; not overwritten via CameraInfo)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fx', 655.68563818)
        self.declare_parameter('fy', 656.8373932213)
        self.declare_parameter('cx', 630.0668311460)
        self.declare_parameter('cy', 366.3269405158)

        # --- NEW: debug vector drawing params ---
        self.declare_parameter('draw_ray3d', True)
        self.declare_parameter('draw_ray2d', True)
        self.declare_parameter('ray_max_len', 50.0)  # fallback length when no ground hit

        # --- Load params ---
        self.bbox_topic     = self.get_parameter('bbox_topic').value
        self.monitor_topic  = self.get_parameter('monitor_topic').value
        self.cls_filter     = self.get_parameter('class_filter').value.strip().lower()

        cam_roll_deg = float(self.get_parameter('cam_roll_deg').value)
        cam_pitch_deg = float(self.get_parameter('cam_pitch_deg').value)
        self.D_ground = float(self.get_parameter('D_ground').value)

        self.field_pos_x = self.get_parameter('field_pos_x').value
        self.field_pos_y = self.get_parameter('field_pos_y').value
        self.field_pos_z = self.get_parameter('field_pos_z').value
        self.field_heading = self.get_parameter('field_heading_rad').value  # radians

        # Camera intrinsics
        self.W = int(self.get_parameter('width').value)
        self.H = int(self.get_parameter('height').value)
        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        self.K = np.array([[fx, 0.0, cx],
                           [0.0, fy, cy],
                           [0.0, 0.0, 1.0]], dtype=float)

        # NEW: vector draw settings
        self.draw_ray3d = bool(self.get_parameter('draw_ray3d').value)
        self.draw_ray2d = bool(self.get_parameter('draw_ray2d').value)
        self.ray_max_len = float(self.get_parameter('ray_max_len').value)

        # --- State ---
        self.C_ned = np.zeros(3, dtype=float)      # drone position [N,E,D]
        self.heading_rad = 0.0                     # drone heading(rad)
        # Camera-to-body fixed rotation (yaw=0)
        self.R_bc_fixed = Rz(0.0) @ Ry(math.radians(cam_pitch_deg)) @ Rx(math.radians(cam_roll_deg))

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
        self.create_subscription(Yolov8Inference, self.bbox_topic, self.cb_bboxes, reliable_qos)
        self.create_subscription(Monitoring, self.monitor_topic, self.cb_monitor, best_effort)

        # --- Pubs ---
        self.pub_points  = self.create_publisher(Marker, '/ground_objects/tag', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/ground_objects/markers', 10)
        self.pub_drone_marker = self.create_publisher(Marker, '/ground_objects/drone_marker', 10)
        self.pub_vectors = self.create_publisher(MarkerArray, '/ground_objects/vectors', 10)  # NEW

        self.get_logger().info(
            f'started. bbox_topic={self.bbox_topic}, K=({fx:.3f},{fy:.3f},{cx:.3f},{cy:.3f}), '
            f'size={self.W}x{self.H}, cam(pitch,roll)=({cam_pitch_deg:.1f},{cam_roll_deg:.1f}) deg'
        )
        self.roll = None
        self.pitch = None
        self.yaw = None

    # ---------- Callbacks ----------
    def cb_monitor(self, msg: Monitoring):
        # Drone position [N,E,D] and heading(rad)
        try:
            self.C_ned[0] = float(getattr(msg, self.field_pos_x))
            self.C_ned[1] = float(getattr(msg, self.field_pos_y))
            self.C_ned[2] = float(getattr(msg, self.field_pos_z))
            self.roll = msg.roll
            self.pitch = msg.pitch
            self.yaw = msg.head
            
        except Exception:
            return
        try:
            self.heading_rad = float(getattr(msg, self.field_heading))
        except Exception:
            self.get_logger().warn("Heading warning")
            self.heading_rad = 0.0

        self._publish_drone_marker()

    def _publish_drone_marker(self):
        m = Marker()
        m.header.frame_id = 'inference'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'drone'
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        # NOTE: This uses ENU-like mapping (x=E, y=N, z=Up=-D) as you had it.
        # If you want consistency with vectors/points below (which use NED numbers),
        # change to: x=N, y=E, z=D_ground or z=D as needed.
        m.pose.position.x = float(self.C_ned[1])
        m.pose.position.y = float(self.C_ned[0])
        m.pose.position.z = -float(self.C_ned[2])
        
        r_ned = R.from_euler('xyz', [self.roll, self.pitch, self.yaw])
        offset = R.from_euler('z', -np.pi/2, degrees=False)
        q = (r_ned * offset).as_quat()
        m.pose.orientation.x = q[1]
        m.pose.orientation.y = q[0]
        m.pose.orientation.z = -q[2]
        m.pose.orientation.x = q[3]
        m.scale.x = m.scale.y = m.scale.z = 0.7
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 0.6
        m.color.b = 1.0

        self.pub_drone_marker.publish(m)

    def cb_bboxes(self, msg: Yolov8Inference):
        if self.K is None:
            return

        boxes: List[BBox] = [BBox(it.left, it.top, it.right, it.bottom, it.class_name)
                             for it in msg.yolov8_inference]

        if not boxes:
            # Clear markers
            ma = MarkerArray()
            clear = Marker()
            clear.header = self._derive_header(msg.header)
            clear.ns = 'ground_objects'; clear.id = 0; clear.action = Marker.DELETEALL
            ma.markers.append(clear)
            self.pub_markers.publish(ma)

            # Clear vectors
            va = MarkerArray()
            vclear = Marker()
            vclear.header = self._derive_header(msg.header)
            vclear.ns = 'vectors'; vclear.id = 0; vclear.action = Marker.DELETEALL
            va.markers.append(vclear)
            self.pub_vectors.publish(va)
            return

        header = self._derive_header(msg.header)
        ma = MarkerArray()
        clear = Marker()
        clear.header = header; clear.ns = 'ground_objects'
        clear.id = 0; clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        # NEW: vector array (clear every frame)
        va = MarkerArray()
        vclear = Marker()
        vclear.header = header; vclear.ns = 'vectors'
        vclear.id = 0; vclear.action = Marker.DELETEALL
        va.markers.append(vclear)

        # Camera->NED rotation: R_nc = Rz(heading_rad) @ R_bc_fixed
        R_nc = Rz(self.heading_rad) @ self.R_bc_fixed
        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]

        self.get_logger().info(
            f"[BB] R_nc[0,*]=[{R_nc[0,0]:.3f},{R_nc[0,1]:.3f},{R_nc[0,2]:.3f}], "
            f"R_nc[2,*]=[{R_nc[2,0]:.3f},{R_nc[2,1]:.3f},{R_nc[2,2]:.3f}]"
        )

        mid = 1      # IDs for ground_objects (if you add any spheres there)
        vid = 1      # IDs for vectors

        for b in boxes:
            if self.cls_filter and b.cls != self.cls_filter:
                self.get_logger().info(f"[BB] skip class='{b.cls}' (filter='{self.cls_filter}')")
                continue

            # Use bottom-center of bbox
            u = 0.5 * (b.l + b.r)
            v = min(max(b.b, 0.0), float(self.H - 1))

            x = (u - cx) / fx
            y = (v - cy) / fy
            d_c = np.array([x, y, 1.0], dtype=float)    # camera-frame ray
            d_n = R_nc @ d_c                            # NED ray
            dz = d_n[2]

            self.get_logger().info(
                f"[RAY] cls={b.cls} u={u:.1f} v={v:.1f} -> d_c=[{x:.3f},{y:.3f},1.000] "
                f"d_n=[{d_n[0]:.3f},{d_n[1]:.3f},{d_n[2]:.3f}]"
            )
            self.get_logger().info("\n")

            if abs(dz) < 1e-9:
                self.get_logger().warn(f"[RAY] dz≈0 -> skip (parallel to ground)")
                self.get_logger().info("\n")
                # Even if parallel, draw a short debug 3D ray if enabled
                if self.draw_ray3d:
                    start = Point(x=float(self.C_ned[1]), y=float(self.C_ned[0]), z=-float(self.C_ned[2]))
                    end3 = self.C_ned + (self.ray_max_len * (d_n / (np.linalg.norm(d_n) + 1e-9)))
                    end = Point(x=float(end3[0]), y=float(end3[1]), z=float(end3[2]))
                    arrow = Marker()
                    arrow.header = header
                    arrow.ns = 'ray3d'; arrow.id = vid; vid += 1
                    arrow.type = Marker.ARROW; arrow.action = Marker.ADD
                    arrow.points = [start, end]
                    arrow.scale.x = 0.05  # shaft dia
                    arrow.scale.y = 0.12  # head dia
                    arrow.scale.z = 0.20  # head len
                    arrow.color.a = 1.0; arrow.color.r = 1.0; arrow.color.g = 0.0; arrow.color.b = 0.0  # red
                    va.markers.append(arrow)
                continue

            # Ground-plane intersection
            t = (self.D_ground - self.C_ned[2]) / dz
            if t <= 0:
                self.get_logger().warn(
                    f"[RAY] t<=0 -> skip (Dg={self.D_ground:.3f}, Cz={self.C_ned[2]:.3f}, dz={dz:.3f}, t={t:.3f})"
                )
                self.get_logger().info("\n")
                # Draw a short debug 3D ray forward to visualize direction
                if self.draw_ray3d:
                    start = Point(x=float(self.C_ned[1]), y=float(self.C_ned[0]), z=-float(self.C_ned[2]))
                    end3 = self.C_ned + (min(self.ray_max_len, 10.0) * (d_n / (np.linalg.norm(d_n) + 1e-9)))
                    end = Point(x=float(end3[0]), y=float(end3[1]), z=float(end3[2]))
                    arrow = Marker()
                    arrow.header = header
                    arrow.ns = 'ray3d'; arrow.id = vid; vid += 1
                    arrow.type = Marker.ARROW; arrow.action = Marker.ADD
                    arrow.points = [start, end]
                    arrow.scale.x = 0.05
                    arrow.scale.y = 0.12
                    arrow.scale.z = 0.20
                    arrow.color.a = 1.0; arrow.color.r = 1.0; arrow.color.g = 0.0; arrow.color.b = 0.0  # red
                    va.markers.append(arrow)
                continue

            # Valid hit point on ground plane
            P = self.C_ned + t * d_n  # [N,E,D_ground]
            self.get_logger().info(
                f"[HIT] P=[{P[0]:.3f},{P[1]:.3f},{P[2]:.3f}] "
                f"(C=[{self.C_ned[0]:.3f},{self.C_ned[1]:.3f},{self.C_ned[2]:.3f}], t={t:.3f})"
            )
            self.get_logger().info("\n")

            # Publish hit point sphere (as before)
            m = Marker()
            m.header.frame_id = 'inference'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'point'
            m.id = 2
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(P[0])
            m.pose.position.y = float(P[1])
            m.pose.position.z = float(self.D_ground)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.7
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 0.6
            m.color.b = 0.0
            self.pub_points.publish(m)

            # -------- NEW: 3D ray arrow C -> P --------
            if self.draw_ray3d:
                start = Point(x=float(self.C_ned[1]), y=float(self.C_ned[0]), z=-float(self.C_ned[2]))
                end = Point(x=float(P[0]), y=float(P[1]), z=float(P[2]))
                arrow = Marker()
                arrow.header = header
                arrow.ns = 'ray3d'; arrow.id = vid; vid += 1
                arrow.type = Marker.ARROW; arrow.action = Marker.ADD
                arrow.points = [start, end]  # start->end arrow
                arrow.scale.x = 0.05   # shaft diameter
                arrow.scale.y = 0.12   # head diameter
                arrow.scale.z = 0.20   # head length
                arrow.color.a = 1.0
                arrow.color.r = 0.0
                arrow.color.g = 1.0
                arrow.color.b = 0.2    # green-ish
                va.markers.append(arrow)

            # -------- NEW: 2D ground-projected line (C at ground -> P) --------
            if self.draw_ray2d:
                p0 = Point(x=float(self.C_ned[1]), y=float(self.C_ned[0]), z=-float(self.D_ground))
                p1 = Point(x=float(P[0]), y=float(P[1]), z=float(self.D_ground))
                line = Marker()
                line.header = header
                line.ns = 'ray2d'; line.id = vid; vid += 1
                line.type = Marker.LINE_LIST; line.action = Marker.ADD
                line.scale.x = 0.05
                line.color.a = 1.0
                line.color.r = 1.0
                line.color.g = 0.0
                line.color.b = 1.0     # magenta
                line.points = [p0, p1]  # LINE_LIST expects pairs
                va.markers.append(line)

        # Publish arrays
        self.pub_markers.publish(ma)
        self.pub_vectors.publish(va)

    # ---------- Utils ----------
    def _derive_header(self, src_header: Header) -> Header:
        h = Header()
        try:
            h = src_header
        except Exception:
            pass
        if not h.frame_id:
            h.frame_id = 'ned'
        return h

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
