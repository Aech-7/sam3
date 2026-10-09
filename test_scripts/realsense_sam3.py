
import cv2
import torch
import numpy as np
from PIL import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class RealSenseSAM3Node(Node):
    def __init__(self):
        super().__init__("realsense_sam3")

        self.bridge = CvBridge()
        self.last_run = 0.0
        self.inference_interval = 0.5  # Start at 2 FPS

        self.get_logger().info("Loading SAM 3...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self.get_logger().info("SAM 3 loaded.")

        self.subscription = self.create_subscription(
            RosImage,
            "/camera/camera/color/image_raw",
            self.image_callback,
            1,
        )

        self.get_logger().info("Waiting for RealSense RGB frames...")

    def image_callback(self, msg):
        # Limit inference rate; don't process every camera frame.
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_run < self.inference_interval:
            return
        self.last_run = now

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="rgb8"
            )
            image = Image.fromarray(frame)

            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ):
                    state = self.processor.set_image(image)
                    output = self.processor.set_text_prompt(
                        state=state,
                        prompt="bottle",
                    )

            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            masks = output["masks"]

            for mask in masks:
                mask_np = mask.squeeze().detach().float().cpu().numpy()
                mask_np = cv2.resize(
                    mask_np,
                    (display.shape[1], display.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                binary = mask_np > 0.5

                # Overlay the mask on the live frame.
                overlay = display.copy()
                overlay[binary] = (0, 255, 0)
                display = cv2.addWeighted(
                    display, 0.6, overlay, 0.4, 0
                )

            cv2.imshow("RealSense + SAM 3", display)
            cv2.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = RealSenseSAM3Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
