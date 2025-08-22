#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

class CompressedToRaw(Node):
    def __init__(self):
        super().__init__('compressed_to_raw')
        self.declare_parameter('in_topic', '/compressed_image_topic')
        self.declare_parameter('out_topic', '/recon_1/camera/image_raw')
        self.declare_parameter('reliability', 'best_effort')  # 'best_effort' or 'reliable'

        in_topic  = self.get_parameter('in_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('out_topic').get_parameter_value().string_value
        reliability = self.get_parameter('reliability').get_parameter_value().string_value

        qos = QoSProfile(
            reliability=(ReliabilityPolicy.RELIABLE if reliability == 'reliable'
                         else ReliabilityPolicy.BEST_EFFORT),
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, out_topic, qos)
        self.sub = self.create_subscription(CompressedImage, in_topic, self.cb, qos)
        self.get_logger().info(f'in: {in_topic} -> out: {out_topic}, reliability={reliability}')

    def cb(self, msg: CompressedImage):
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)   # BGR
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            img_msg.header = msg.header
            self.pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'convert failed: {e}')

def main():
    rclpy.init()
    node = CompressedToRaw()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
