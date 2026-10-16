#!/usr/bin/env python3
"""Offline tests for the same pure functions used by the ROS nodes."""

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SCRIPT_DIR)

from drawbot_core import (
    SketchGameState,
    build_prediction_payload,
    confidence_message,
    parse_voice_command,
)
from standalone_demo import create_synthetic_camera_sketch, preprocess_frame_stages
from train_drawing_classifier import inspect_h5_model_config


class TestDrawBotCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(PKG_ROOT, "models", "drawing_classifier.h5")
        cls.classes_path = os.path.join(PKG_ROOT, "config", "classes.txt")
        with open(cls.classes_path, encoding="utf-8") as handle:
            cls.classes = [line.strip() for line in handle if line.strip()]

    def test_voice_parser_uses_production_logic(self):
        examples = {
            "start the game": "start",
            "I'm ready": "ready",
            "please recognize it": "ready",
            "one more time": "repeat",
            "stop the game": "stop",
            "unrelated words": None,
        }
        for transcript, expected in examples.items():
            with self.subTest(transcript=transcript):
                self.assertEqual(parse_voice_command(transcript), expected)
        self.assertEqual(parse_voice_command("start and then stop"), "stop")

    def test_state_machine_guards_recognition(self):
        state = SketchGameState()
        message, recognize = state.handle("ready")
        self.assertEqual(message, "Please start the game first.")
        self.assertFalse(recognize)
        state.handle("start")
        _, recognize = state.handle("ready")
        self.assertTrue(recognize)
        state.handle("stop")
        self.assertFalse(state.started)

    def test_prediction_payload_and_confidence_messages(self):
        probabilities = np.zeros(len(self.classes), dtype=float)
        probabilities[self.classes.index("basketball")] = 0.924
        probabilities[self.classes.index("baseball")] = 0.045
        probabilities[self.classes.index("apple")] = 0.018
        payload = build_prediction_payload(probabilities, self.classes)
        self.assertEqual(payload["top1"]["label"], "basketball")
        self.assertEqual(confidence_message(payload), "I think this is a basketball.")

        probabilities[:] = 0
        probabilities[self.classes.index("cat")] = 0.54
        probabilities[self.classes.index("dog")] = 0.22
        probabilities[self.classes.index("tiger")] = 0.11
        medium = build_prediction_payload(probabilities, self.classes)
        self.assertEqual(
            confidence_message(medium),
            "I am not very sure. It may be cat, dog, tiger.",
        )

        probabilities[:] = 0
        probabilities[self.classes.index("line")] = 0.21
        low = build_prediction_payload(probabilities, self.classes)
        self.assertIn("draw it again", confidence_message(low))

    def test_prediction_payload_validates_dimensions(self):
        with self.assertRaises(ValueError):
            build_prediction_payload([0.1, 0.9], self.classes)

    def test_preprocessing_contract(self):
        frame = create_synthetic_camera_sketch("basketball")
        tensor, stages = preprocess_frame_stages(frame)
        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertEqual(len(stages), 6)

    def test_bundled_model_topology_not_accuracy(self):
        config = inspect_h5_model_config(self.model_path)
        layers = config["config"]["layers"]
        self.assertEqual(layers[0]["config"]["batch_input_shape"], [None, 28, 28, 1])
        self.assertEqual(layers[-1]["config"]["units"], 345)
        self.assertEqual(len(self.classes), 345)

    def test_privacy_and_process_safety_defaults(self):
        game_path = os.path.join(SCRIPT_DIR, "sketch_game_node.py")
        voice_path = os.path.join(SCRIPT_DIR, "voice_natural_command_node.py")
        with open(game_path, encoding="utf-8") as handle:
            game_source = handle.read()
        with open(voice_path, encoding="utf-8") as handle:
            voice_source = handle.read()
        self.assertNotIn('"pkill"', game_source)
        self.assertNotIn('cv2.imwrite("/tmp', game_source)
        self.assertIn('rospy.get_param("~allow_cloud_stt", False)', voice_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
