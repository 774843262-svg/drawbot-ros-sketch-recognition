#!/usr/bin/env python3
"""Compatible CNN definition, HDF5 inspector and optional training utility.

The topology below is embedded in ``models/drawing_classifier.h5``. The exact
training run that produced the bundled weights is not preserved; see the root
MODEL_CARD.md for the provenance and evaluation boundary.

Bundled model topology:
  1. InputLayer:   (None, 28, 28, 1) float32 grayscale normalized in [0, 1]
  2. Conv2D:       32 filters, 5x5 kernel, padding='same', activation='relu'  -> (None, 28, 28, 32)
  3. MaxPooling2D: 2x2 pool_size, strides=2                                   -> (None, 14, 14, 32)
  4. Conv2D:       32 filters, 3x3 kernel, padding='same', activation='relu'  -> (None, 14, 14, 32)
  5. MaxPooling2D: 2x2 pool_size, strides=2                                   -> (None, 7, 7, 32)
  6. Conv2D:       64 filters, 3x3 kernel, padding='same', activation='relu'  -> (None, 7, 7, 64)
  7. MaxPooling2D: 2x2 pool_size, strides=2                                   -> (None, 3, 3, 64)
  8. Flatten:      576 units                                                  -> (None, 576)
  9. Dense:        512 units, activation='relu'                               -> (None, 512)
 10. Dense:        345 units, activation='softmax'                            -> (None, 345)
"""

import argparse
import glob
import json
import os
import re
import numpy as np


def inspect_h5_model_config(h5_path: str) -> dict:
    """Extracts the embedded Keras JSON model configuration from an HDF5 (.h5) file."""
    with open(h5_path, "rb") as f:
        header_bytes = f.read(65536)
    text = header_bytes.decode("latin1")
    idx = text.find('{"class_name": "Sequential"')
    if idx == -1:
        raise ValueError(f"Could not locate Keras Sequential config in {h5_path}")
    # Parse balanced JSON braces
    depth = 0
    end_idx = idx
    for i, ch in enumerate(text[idx:], start=idx):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    return json.loads(text[idx:end_idx])


def build_quickdraw_cnn(num_classes: int = 345, input_shape=(28, 28, 1)):
    """Builds the Keras Sequential CNN model used in DrawBot."""
    import tensorflow as tf
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape, name="conv2d_57_input"),
            tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv2d_57"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_57"),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_58"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_58"),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_59"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_59"),
            tf.keras.layers.Flatten(name="flatten_19"),
            tf.keras.layers.Dense(512, activation="relu", name="dense_43"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_44"),
        ],
        name="sequential_19",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_quickdraw_npy_folder(data_dir: str, classes: list, max_samples_per_class: int = 4000):
    """Loads Google Quick, Draw! numpy bitmap files (.npy) for training."""
    X_list, y_list = [], []
    for idx, cls_name in enumerate(classes):
        npy_file = os.path.join(data_dir, f"{cls_name}.npy")
        if not os.path.exists(npy_file):
            continue
        arr = np.load(npy_file)[:max_samples_per_class]
        arr = arr.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        X_list.append(arr)
        y_list.append(np.full(len(arr), idx, dtype=np.int32))
    if not X_list:
        raise FileNotFoundError(f"No QuickDraw .npy files found in {data_dir}")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Inspect or train the DrawBot 345-class QuickDraw CNN.")
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.path.dirname(__file__), "..", "models", "drawing_classifier.h5"),
        help="Path to drawing_classifier.h5",
    )
    parser.add_argument(
        "--classes-path",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "classes.txt"),
        help="Path to classes.txt (345 categories)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional directory containing Google Quick, Draw! .npy files to train from scratch",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    classes_path = os.path.abspath(args.classes_path)

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(classes)} QuickDraw categories from {classes_path}")

    if os.path.exists(model_path) and args.data_dir is None:
        cfg = inspect_h5_model_config(model_path)
        print(f"Inspected existing model file at: {model_path}")
        print(f"Model Name: {cfg['config']['name']}")
        print("Layer Topology:")
        for i, layer in enumerate(cfg["config"]["layers"], start=1):
            lname = layer["config"].get("name", "")
            ltype = layer["class_name"]
            extra = ""
            if ltype == "Conv2D":
                extra = f"filters={layer['config']['filters']}, kernel={layer['config']['kernel_size']}, act={layer['config']['activation']}"
            elif ltype == "Dense":
                extra = f"units={layer['config']['units']}, act={layer['config']['activation']}"
            elif ltype == "MaxPooling2D":
                extra = f"pool_size={layer['config']['pool_size']}"
            print(f"  [{i:02d}] {ltype:<15} ({lname:<18}) {extra}")
        return

    if args.data_dir:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        X, y = load_quickdraw_npy_folder(args.data_dir, classes)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        model = build_quickdraw_cnn(num_classes=len(classes))
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        model.save(model_path)
        print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
