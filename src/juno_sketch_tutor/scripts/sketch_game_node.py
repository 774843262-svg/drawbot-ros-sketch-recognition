#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
import tensorflow as tf
import rospy
import rospkg
import subprocess

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from drawbot_core import (
    SketchGameState,
    build_prediction_payload,
    confidence_message,
)
from drawbot_vision import preprocess_sketch_stages


class SketchGameNode:
    def __init__(self):
        # 必须最先初始化 ROS node
        rospy.init_node("sketch_game_node", anonymous=False)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.game_state = SketchGameState()
        self._tts_process = None

        # 读取 ROS 参数
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.use_espeak = rospy.get_param("~use_espeak", True)
        self.debug_dir = os.path.expanduser(rospy.get_param("~debug_dir", ""))
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            rospy.logwarn("Debug frame capture enabled: %s", self.debug_dir)
        rospy.on_shutdown(self.stop_tts)

        # 获取 package 路径
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("juno_sketch_tutor")

        default_model_path = os.path.join(
            pkg_path,
            "models",
            "drawing_classifier.h5"
        )

        default_classes_path = os.path.join(
            pkg_path,
            "config",
            "classes.txt"
        )

        # ROS parameters may override these with local model/class paths.
        self.model_path = rospy.get_param("~model_path", default_model_path)
        self.classes_path = rospy.get_param("~classes_path", default_classes_path)

        # 解决 Python 不自动展开 ~ 的问题
        self.model_path = os.path.expanduser(self.model_path)
        self.classes_path = os.path.expanduser(self.classes_path)

        rospy.loginfo("Package path: {}".format(pkg_path))
        rospy.loginfo("Image topic: {}".format(self.image_topic))
        rospy.loginfo("Model path: {}".format(self.model_path))
        rospy.loginfo("Classes path: {}".format(self.classes_path))

        # 检查模型文件
        if not os.path.exists(self.model_path):
            rospy.logerr("Model file does not exist: {}".format(self.model_path))
            raise FileNotFoundError(self.model_path)

        # 检查 classes.txt
        if not os.path.exists(self.classes_path):
            rospy.logerr("Classes file does not exist: {}".format(self.classes_path))
            raise FileNotFoundError(self.classes_path)

        # 加载模型
        rospy.loginfo("Loading CNN model...")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        rospy.loginfo("Model loaded successfully.")

        # 加载类别
        self.classes = self.load_classes(self.classes_path)

        rospy.loginfo("Classes: {}".format(self.classes))
        rospy.loginfo("Model input shape: {}".format(self.model.input_shape))
        rospy.loginfo("Model output shape: {}".format(self.model.output_shape))

        output_dim = self.model.output_shape[-1]

        if output_dim != len(self.classes):
            rospy.logwarn(
                "Model output classes ({}) != classes.txt length ({}). "
                "Check class order and class count.".format(
                    output_dim,
                    len(self.classes)
                )
            )

        # Publisher
        self.answer_pub = rospy.Publisher(
            "/answer_text",
            String,
            queue_size=10
        )

        self.pred_pub = rospy.Publisher(
            "/predicted_drawing",
            String,
            queue_size=10
        )

        # Subscriber
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        self.command_sub = rospy.Subscriber(
            "/game_command",
            String,
            self.command_callback,
            queue_size=10
        )

        rospy.loginfo("Sketch game node started.")
        rospy.loginfo("Subscribing image topic: {}".format(self.image_topic))
        rospy.loginfo("Waiting for command: start / ready / repeat / stop")

    def load_classes(self, path):
        with open(path, "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]

        if len(classes) == 0:
            raise ValueError("classes.txt is empty.")

        return classes
    def stop_tts(self):
        """Stop only the eSpeak process started by this node."""
        if self._tts_process is not None and self._tts_process.poll() is None:
            self._tts_process.terminate()
            try:
                self._tts_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._tts_process.kill()
        self._tts_process = None

    def speak(self, text):
        rospy.loginfo("Robot says: {}".format(text))
        self.answer_pub.publish(text)

        if self.use_espeak:
            try:
                self.stop_tts()
                self._tts_process = subprocess.Popen([
                    "espeak",
                    "-s", "158",   # speed: lower = slower, default around 175
                    "-a", "120",   # amplitude/volume: 0-200
                    "-g", "5",     # word gap, larger = more pause between words
                    text
                ])
            except Exception as e:
                rospy.logerr("TTS error: {}".format(e))

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            rospy.logerr("Image conversion failed: {}".format(e))

    def get_model_input_info(self):
        input_shape = self.model.input_shape

        if len(input_shape) == 4:
            target_h = input_shape[1] if input_shape[1] is not None else 28
            target_w = input_shape[2] if input_shape[2] is not None else 28
            channels = input_shape[3] if input_shape[3] is not None else 1
            return target_h, target_w, channels, "image"

        elif len(input_shape) == 2:
            # 常见 flatten 输入，例如 784
            flat_dim = input_shape[1]
            if flat_dim is None:
                target_h = 28
                target_w = 28
            else:
                side = int(np.sqrt(flat_dim))
                if side * side == flat_dim:
                    target_h = side
                    target_w = side
                else:
                    target_h = 28
                    target_w = 28

            return target_h, target_w, 1, "flat"

        else:
            raise ValueError("Unsupported model input shape: {}".format(input_shape))

    def preprocess_frame(self, frame):
        """
        摄像头画面预处理：
        目标格式：
        - 黑色背景
        - 白色线条
        - 图形居中
        - 尺寸匹配模型输入
        """

        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "juno_raw_frame.jpg"), frame)

        target_h, target_w, channels, input_mode = self.get_model_input_info()
        resized, _ = preprocess_sketch_stages(frame, target_h=target_h, target_w=target_w)

        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "juno_processed_sketch.png"), resized)

        x = resized.astype("float32") / 255.0

        if input_mode == "image":
            if channels == 1:
                x = x.reshape(1, target_h, target_w, 1)

            elif channels == 3:
                x = np.stack([x, x, x], axis=-1)
                x = x.reshape(1, target_h, target_w, 3)

            else:
                raise ValueError("Unsupported channel number: {}".format(channels))

        elif input_mode == "flat":
            x = x.reshape(1, target_h * target_w)

        else:
            raise ValueError("Unsupported input mode: {}".format(input_mode))

        return x

    def recognize_current_frame(self):
        if self.latest_frame is None:
            self.speak("I cannot see the camera image yet.")
            return

        try:
            x = self.preprocess_frame(self.latest_frame)

            pred = self.model.predict(x, verbose=0)[0]

            result = build_prediction_payload(pred, self.classes, top_k=3)

            result_json = json.dumps(result)
            self.pred_pub.publish(result_json)
            rospy.loginfo("Prediction result: {}".format(result_json))

            self.speak(confidence_message(result))

        except Exception as e:
            rospy.logerr("Recognition failed: {}".format(e))
            self.speak("Recognition failed. Please check the camera image and model input.")

    def command_callback(self, msg):
        command = msg.data.strip().lower()
        rospy.loginfo("Received command: {}".format(command))

        message, should_recognize = self.game_state.handle(command)
        self.speak(message)
        if should_recognize:
            rospy.sleep(0.5)
            self.recognize_current_frame()


if __name__ == "__main__":
    try:
        node = SketchGameNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
