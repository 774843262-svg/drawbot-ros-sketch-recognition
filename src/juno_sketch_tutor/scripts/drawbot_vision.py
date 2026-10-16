#!/usr/bin/env python3
"""ROS-independent OpenCV preprocessing shared by runtime and tests."""

import cv2
import numpy as np


def preprocess_sketch_stages(
    frame: np.ndarray, target_h: int = 28, target_w: int = 28
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the resized grayscale sketch and inspectable intermediate stages."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape (height, width, 3)")
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target dimensions must be positive")

    height, width, _ = frame.shape
    side = min(height, width)
    x_start = (width - side) // 2
    y_start = (height - side) // 2
    roi = frame[y_start:y_start + side, x_start:x_start + side]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped = morph
    if contours:
        image_area = morph.shape[0] * morph.shape[1]
        mask = np.zeros_like(morph)
        for contour in contours:
            if cv2.contourArea(contour) > image_area * 0.0005:
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        ys, xs = np.where(mask > 0)
        if len(xs) and len(ys):
            pad = 20
            x_min = max(0, int(np.min(xs)) - pad)
            y_min = max(0, int(np.min(ys)) - pad)
            x_max = min(morph.shape[1], int(np.max(xs)) + pad + 1)
            y_max = min(morph.shape[0], int(np.max(ys)) + pad + 1)
            cropped = morph[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        cropped = morph

    crop_h, crop_w = cropped.shape
    square_size = max(crop_h, crop_w)
    square = np.zeros((square_size, square_size), dtype=np.uint8)
    y_offset = (square_size - crop_h) // 2
    x_offset = (square_size - crop_w) // 2
    square[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w] = cropped
    resized = cv2.resize(square, (target_w, target_h), interpolation=cv2.INTER_AREA)

    stages = {
        "1_raw_camera_frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        "2_center_crop_gray": gray,
        "3_otsu_binary_inv": binary,
        "4_morph_cleaned": morph,
        "5_contour_square_pad": square,
        "6_resized_model_input": resized,
    }
    return resized, stages
