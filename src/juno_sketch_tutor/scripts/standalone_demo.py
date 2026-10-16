#!/usr/bin/env python3
"""Generate an inspectable preprocessing example without ROS or hardware."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from drawbot_vision import preprocess_sketch_stages


def create_synthetic_camera_sketch(shape_type: str = "basketball") -> np.ndarray:
    """Create a synthetic 640×480 paper-and-ink camera frame."""
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    yy, xx = np.mgrid[0:480, 0:640]
    lighting = ((xx - 320) * 0.02 + (yy - 240) * 0.02).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + lighting[:, :, None], 200, 250).astype(np.uint8)

    center_x, center_y = 320, 240
    if shape_type == "basketball":
        cv2.circle(frame, (center_x, center_y), 95, (25, 25, 25), 6, cv2.LINE_AA)
        cv2.line(frame, (center_x - 95, center_y), (center_x + 95, center_y), (25, 25, 25), 5, cv2.LINE_AA)
        cv2.line(frame, (center_x, center_y - 95), (center_x, center_y + 95), (25, 25, 25), 5, cv2.LINE_AA)
        cv2.ellipse(frame, (center_x - 50, center_y), (65, 85), 0, -65, 65, (25, 25, 25), 5, cv2.LINE_AA)
        cv2.ellipse(frame, (center_x + 50, center_y), (65, 85), 180, -65, 65, (25, 25, 25), 5, cv2.LINE_AA)
    elif shape_type == "cup":
        cv2.rectangle(frame, (center_x - 60, center_y - 55), (center_x + 45, center_y + 75), (25, 25, 25), 6, cv2.LINE_AA)
        cv2.ellipse(frame, (center_x + 45, center_y + 10), (35, 42), 0, -90, 90, (25, 25, 25), 6, cv2.LINE_AA)
    elif shape_type == "faint_ambiguous":
        cv2.line(frame, (center_x - 40, center_y - 20), (center_x + 35, center_y + 15), (185, 185, 185), 1, cv2.LINE_AA)
        cv2.line(frame, (center_x - 10, center_y + 30), (center_x + 25, center_y - 25), (185, 185, 185), 1, cv2.LINE_AA)
    else:
        raise ValueError(f"Unknown synthetic shape: {shape_type}")
    return frame


def preprocess_frame_stages(frame: np.ndarray, target_h: int = 28, target_w: int = 28):
    """Compatibility wrapper returning a model tensor plus shared runtime stages."""
    resized, stages = preprocess_sketch_stages(frame, target_h, target_w)
    tensor = (resized.astype("float32") / 255.0).reshape(1, target_h, target_w, 1)
    return tensor, stages


def save_preprocessing_figure(output_path: str) -> None:
    frame = create_synthetic_camera_sketch("basketball")
    _, stages = preprocess_frame_stages(frame)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=150)
    titles = [
        "1. Synthetic camera frame (640×480)",
        "2. Center crop and grayscale",
        "3. Gaussian blur and Otsu threshold",
        "4. Morphological denoising",
        "5. Contour crop and square padding",
        "6. Resized model input (28×28)",
    ]
    for axis, image, title in zip(axes.flat, stages.values(), titles):
        axis.imshow(image if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
        axis.set_title(title, fontsize=9.5, fontweight="bold")
        axis.axis("off")
    fig.suptitle("DrawBot shared OpenCV preprocessing pipeline", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved vision preprocessing pipeline visualization to: {output_path}")


if __name__ == "__main__":
    destination = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "docs", "vision_preprocessing_pipeline.png")
    )
    save_preprocessing_figure(destination)
