# import rclpy
# from rclpy.node import Node
# import cv2
# import torch
# import openvino as ov
# from .modules.groundedsam import ImageProcessor
# from PIL import Image

# class VideoDisplayNode(Node):
#     def __init__(self):
#         super().__init__('video_display_node')
        
#         self.declare_parameter('video_source', '/dev/video0')
#         self.declare_parameter('isSegment', False)
#         self.declare_parameter('detectionList', ['person'])

#         self.video_source = self.get_parameter('video_source').value
#         self.isSegment = self.get_parameter('isSegment').value
#         self.detectionList = self.get_parameter('detectionList').value

#         self.get_logger().info(f'Video source: {self.video_source}')
#         self.get_logger().info(f'isSegment: {self.isSegment}')
#         self.get_logger().info('My list:')
#         for item in self.detectionList:
#             self.get_logger().info(f' - {item}')

#         self.cap = cv2.VideoCapture(self.video_source)

#         self.core = ov.Core()
#         self.device = "CPU" #"GPU.0"
#         self.processor = ImageProcessor(self.core, self.device)

#     def __del__(self):
#         self.cap.release()

#     def run(self):
#         ret, frame = self.cap.read()
#         if ret:
#             self.pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             self.processed_image = self.processor.process_image(self.pil_image, self.detectionList, self.isSegment)
#             cv2.imshow('Video Frame', self.processed_image)
#             cv2.waitKey(0)

#     # def run(self):
#     #     self.get_logger().info("Starting live inference... Saving to inference_output.mp4")
        
#     #     # Get frame dimensions to initialize the video writer
#     #     ret, frame = self.cap.read()
#     #     if not ret:
#     #         self.get_logger().error("Could not read from video source")
#     #         return

#     #     height, width, _ = frame.shape
#     #     fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
#     #     out = cv2.VideoWriter('inference_output.mp4', fourcc, 10.0, (width, height))

#     #     try:
#     #         while rclpy.ok():
#     #             ret, frame = self.cap.read()
#     #             if not ret:
#     #                 break
                
#     #             # AI Processing
#     #             pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     #             processed_image = self.processor.process_image(pil_image, self.detectionList, self.isSegment)
                
#     #             # Save the frame to file
#     #             out.write(processed_image)
                
#     #             # Optional: still try to show window (will fail gracefully if no display)
#     #             # cv2.imshow('Grounded SAM Live', processed_image)
#     #             # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #             #     break
                
#     #             self.get_logger().info("Processed 1 frame...", once=False)

#     #     except Exception as e:
#     #         self.get_logger().error(f"Error during inference: {e}")
#     #     finally:
#     #         out.release()
#     #         self.cap.release()
#     #         cv2.destroyAllWindows()
#     #         self.get_logger().info("Video saved successfully.")

#     #     cv2.destroyAllWindows()

# def main(args=None):
#     rclpy.init(args=args)
#     node = VideoDisplayNode()
#     node.run()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# import cv2
# import numpy as np
# import openvino as ov
# from PIL import Image as PILImage
# from .modules.groundedsam import ImageProcessor

# class VideoInferenceNode(Node):
#     def __init__(self):
#         super().__init__('video_inference_node')
        
#         # Parameters
#         self.declare_parameter('isSegment', False)
#         self.declare_parameter('detectionList', ['person'])
        
#         self.isSegment = self.get_parameter('isSegment').value
#         self.detectionList = self.get_parameter('detectionList').value

#         # Initialize AI Processor
#         self.core = ov.Core()
#         self.processor = ImageProcessor(self.core, "CPU")
        
#         # Initialize CV Bridge
#         self.bridge = CvBridge()

#         # --- FIX: QoS Profile to prevent buffer lag ---
#         # This tells ROS to throw away old frames and only keep the 1 newest frame.
#         qos_profile = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1
#         )

#         # Create Subscriber (Gazebo Camera) with QoS
#         self.subscription = self.create_subscription(
#             Image,
#             '/camera/image',
#             self.image_callback,
#             qos_profile)
        
#         # Create Publisher (Detection Result)
#         self.publisher = self.create_publisher(Image, '/detection/image', 10)

#         self.get_logger().info(f"Grounded-SAM started. Looking for: {self.detectionList}")

#     def image_callback(self, msg):
#         try:
#             # 1. Convert ROS Image to OpenCV BGR
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

#             # --- OPTIMIZATION: Downscale image for faster CPU processing ---
#             # Processing 640x480 is much faster than high-res Gazebo output
#             height, width = cv_image.shape[:2]
#             if width > 640:
#                 cv_image = cv2.resize(cv_image, (640, 480))

#             # 2. Convert to PIL for the Processor
#             pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

#             # 3. Run Inference
#             self.get_logger().info("Processing frame...", once=False)
#             processed_image = self.processor.process_image(
#                 pil_image, 
#                 self.detectionList, 
#                 self.isSegment
#             )

#             # 4. Convert back to ROS Image and Publish
#             out_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
#             out_msg.header = msg.header  # Preserve timestamp from Gazebo
#             self.publisher.publish(out_msg)
            
#             self.get_logger().info("Detection frame published!")

#         except Exception as e:
#             self.get_logger().error(f'Inference failed: {str(e)}')

# def main(args=None):
#     rclpy.init(args=args)
#     node = VideoInferenceNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()



#--only camera working
#!/usr/bin/env python3
# """
# ROS 2 node that:
# - Subscribes to /camera/image
# - Republishes the same Image message to /detection/image

# Purpose:
# - Sanity check that image transport, QoS, and topic wiring work
# - No AI / inference logic yet
# """

# import rclpy
# from rclpy.node import Node

# # Standard ROS 2 Image message
# from sensor_msgs.msg import Image

# # QoS profile optimized for sensor data (camera streams)
# from rclpy.qos import qos_profile_sensor_data


# class ImageRelayNode(Node):
#     def __init__(self):
#         super().__init__('image_relay_node')

#         # Publisher: where the relayed image will be sent
#         self.image_pub = self.create_publisher(
#             Image,
#             '/detection/image',
#             qos_profile_sensor_data
#         )

#         # Subscriber: input camera topic
#         self.image_sub = self.create_subscription(
#             Image,
#             '/camera/image',
#             self.image_callback,
#             qos_profile_sensor_data
#         )

#         self.get_logger().info('Image relay node started.')
#         self.get_logger().info('Subscribing to: /camera/image')
#         self.get_logger().info('Publishing to:  /detection/image')

#     def image_callback(self, msg: Image):
#         """
#         Callback executed every time a new Image message arrives.

#         For now:
#         - Simply republish the same message
#         - No modification, no copying, no inference
#         """

#         # Republish image as-is
#         self.image_pub.publish(msg)

#         # Lightweight debug log (throttled manually by user if needed)
#         self.get_logger().debug(
#             f'Relayed image: {msg.width}x{msg.height}, encoding={msg.encoding}'
#         )


# def main(args=None):
#     # Initialize ROS 2 Python client library
#     rclpy.init(args=args)

#     # Create node instance
#     node = ImageRelayNode()

#     try:
#         # Keep node alive and processing callbacks
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         # Clean shutdown
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()


#---drawing boxes correctly--
# #!/usr/bin/env python3
# """
# ROS 2 AI Image Inference Node

# - Subscribes to /camera/image
# - Runs OpenVINO + GroundedSAM inference
# - Publishes annotated image to /detection/image
# """

# import rclpy
# from rclpy.node import Node

# # ROS image message
# from sensor_msgs.msg import Image

# # QoS for camera streams
# from rclpy.qos import qos_profile_sensor_data

# # Image conversion
# from cv_bridge import CvBridge

# # Computer vision / AI
# import cv2
# import openvino as ov
# from PIL import Image as PILImage

# # Your existing AI module
# from .modules.groundedsam import ImageProcessor


# class ImageInferenceNode(Node):
#     def __init__(self):
#         super().__init__('video_display_node')

#         # -----------------------------
#         # Parameters (same as your code)
#         # -----------------------------
#         self.declare_parameter('isSegment', False)
#         self.declare_parameter('detectionList', ['person'])

#         self.isSegment = self.get_parameter('isSegment').value
#         self.detectionList = self.get_parameter('detectionList').value

#         self.get_logger().info(f'isSegment: {self.isSegment}')
#         self.get_logger().info(f'detectionList: {self.detectionList}')

#         # -----------------------------
#         # ROS interfaces
#         # -----------------------------
#         self.bridge = CvBridge()

#         self.image_sub = self.create_subscription(
#             Image,
#             '/camera/image',
#             self.image_callback,
#             qos_profile_sensor_data
#         )

#         self.image_pub = self.create_publisher(
#             Image,
#             '/detection/image',
#             qos_profile_sensor_data
#         )

#         # -----------------------------
#         # AI initialization (ONCE)
#         # -----------------------------
#         self.core = ov.Core()
#         self.device = "CPU"  # or "GPU.0"
#         self.processor = ImageProcessor(self.core, self.device)

#         self.get_logger().info('Image inference node started.')
#         self.get_logger().info('Waiting for images...')

#     def image_callback(self, msg: Image):
#         """
#         Pipeline:
#         ROS Image → OpenCV → PIL → AI inference → OpenCV → ROS Image
#         """

#         try:
#             # -----------------------------
#             # ROS Image → OpenCV (BGR)
#             # -----------------------------
#             cv_image = self.bridge.imgmsg_to_cv2(
#                 msg,
#                 desired_encoding='bgr8'
#             )

#             # -----------------------------
#             # OpenCV → PIL (RGB)
#             # -----------------------------
#             pil_image = PILImage.fromarray(
#                 cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
#             )

#             # -----------------------------
#             # AI inference
#             # -----------------------------
#             processed_image = self.processor.process_image(
#                 pil_image,
#                 self.detectionList,
#                 self.isSegment
#             )

#             # processed_image is assumed OpenCV BGR
#             # If RGB, convert here accordingly

#             # -----------------------------
#             # OpenCV → ROS Image
#             # -----------------------------
#             out_msg = self.bridge.cv2_to_imgmsg(
#                 processed_image,
#                 encoding='bgr8'
#             )

#             # Preserve header timing/frame
#             out_msg.header = msg.header

#             # -----------------------------
#             # Publish result
#             # -----------------------------
#             self.image_pub.publish(out_msg)

#         except Exception as e:
#             self.get_logger().error(f'Inference error: {e}')


# def main(args=None):
#     rclpy.init(args=args)

#     node = ImageInferenceNode()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
"""
Grounded-SAM ROS 2 Node (Optimized)

- Subscribes: /camera/image
- Runs inference at fixed rate (not per frame)
- Publishes: /detection/image
- Draws bounding boxes + class labels
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import openvino as ov
from PIL import Image as PILImage
import time

from .modules.groundedsam import ImageProcessor


class GroundedSAMNode(Node):
    def __init__(self):
        super().__init__('grounded_sam')

        # ---------------- Parameters ----------------
        self.declare_parameter('device', 'CPU')
        self.declare_parameter('isSegment', False)
        self.declare_parameter('detectionList', ['person'])
        self.declare_parameter('inference_fps', 1.0)  # IMPORTANT

        self.device = self.get_parameter('device').value
        self.isSegment = self.get_parameter('isSegment').value
        self.detectionList = self.get_parameter('detectionList').value
        self.inference_fps = self.get_parameter('inference_fps').value

        self.get_logger().info(f'Device: {self.device}')
        self.get_logger().info(f'isSegment: {self.isSegment}')
        self.get_logger().info(f'Detections: {self.detectionList}')
        self.get_logger().info(f'Inference FPS: {self.inference_fps}')

        # ---------------- ROS I/O ----------------
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_header = None

        self.sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.pub = self.create_publisher(
            Image,
            '/detection/image',
            qos_profile_sensor_data
        )

        # ---------------- AI Init ----------------
        self.core = ov.Core()
        self.processor = ImageProcessor(self.core, self.device)

        # ---------------- Timer ----------------
        period = 1.0 / self.inference_fps
        self.timer = self.create_timer(period, self.run_inference)

        self.get_logger().info('Grounded-SAM node initialized.')

    def image_callback(self, msg: Image):
        """Always keep only the latest frame"""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.latest_header = msg.header

    def run_inference(self):
        """Runs at fixed rate, NOT per frame"""
        if self.latest_image is None:
            return

        start = time.time()

        try:
            # OpenCV → PIL
            pil_image = PILImage.fromarray(
                cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
            )

            # Inference
            output = self.processor.process_image(
                pil_image,
                self.detectionList,
                self.isSegment
            )

            # Publish
            out_msg = self.bridge.cv2_to_imgmsg(output, 'bgr8')
            out_msg.header = self.latest_header
            self.pub.publish(out_msg)

            dt = time.time() - start
            self.get_logger().info(f'Inference time: {dt:.2f}s')

        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = GroundedSAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
