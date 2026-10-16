# DrawBot ROS Sketch Recognition

A course-era human–robot interaction prototype that turns a camera sketch into a 345-class prediction, publishes structured ROS messages and provides uncertainty-aware spoken feedback.

The public edition separates ROS integration from testable Python logic, makes camera debug capture opt-in, disables cloud speech recognition by default, and labels the bundled model's evidence limits.

> Academic team project by Jiang Xianbo, Li Lingyi, Li Chenxue and Du Jiaxue. Student identifiers, submitted reports and terminal screenshots are excluded from this public edition.

## Architecture

```text
USB camera → OpenCV preprocessing → 28×28×1 tensor → bundled CNN
                                                        └─ top-3 JSON → confidence message → eSpeak

optional microphone → Google STT (explicit opt-in) → command parser → game state machine
```

ROS topics:

| Topic | Type | Purpose |
|---|---|---|
| `/usb_cam/image_raw` | `sensor_msgs/Image` | camera frames |
| `/game_command` | `std_msgs/String` | `start`, `ready`, `repeat`, `stop` |
| `/predicted_drawing` | `std_msgs/String` | JSON top-1/top-3 predictions |
| `/answer_text` | `std_msgs/String` | uncertainty-aware response text |

## Engineering changes in the public edition

- `drawbot_core.py` contains the command parser, state machine, prediction payload and confidence messages used by the ROS nodes and imported directly by tests.
- The TTS node tracks and terminates only its own `espeak` process; it no longer issues a system-wide `pkill`.
- Raw and processed camera frames are not written by default. Set `debug_dir` only when the people and environment being recorded have consented.
- Google speech recognition is disabled unless `allow_cloud_stt:=true` is supplied. Enabling it sends captured audio to an external service.
- Offline tests validate preprocessing shape, routing/state behavior, payloads and the bundled HDF5 topology. They do **not** establish classifier accuracy or robot reliability.

## Offline verification (no ROS or TensorFlow required)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt

python -m unittest discover src/juno_sketch_tutor/test -v
python src/juno_sketch_tutor/scripts/standalone_demo.py
python src/juno_sketch_tutor/scripts/train_drawing_classifier.py
```

The final command inspects the embedded model topology without loading TensorFlow.

## ROS launch

This package targets Ubuntu 20.04 and ROS Noetic, the environment used by the course hardware. [ROS Noetic reached end of life on 31 May 2025](https://discourse.ros.org/t/ros-noetic-end-of-life-may-31-2025/43160), so this should be treated as a reproducible legacy prototype; new deployment work should plan a ROS 2 migration.

```bash
catkin_make
source devel/setup.bash

# Camera + classifier; no cloud speech
roslaunch juno_sketch_tutor juno_sketch_game_usb.launch

# Voice node remains idle unless cloud STT is explicitly enabled
roslaunch juno_sketch_tutor juno_sketch_game_voice.launch

# Explicit opt-in: captured speech is sent to Google's recognition service
roslaunch juno_sketch_tutor juno_sketch_game_voice.launch allow_cloud_stt:=true
```

Optional camera diagnostics:

```bash
roslaunch juno_sketch_tutor juno_sketch_game_usb.launch debug_dir:=/path/you/control
```

## Model scope

The bundled `drawing_classifier.h5` exposes a `28×28×1 → 345` Keras CNN topology and is paired with the 345 Quick, Draw! labels. The original training split, checkpoints, evaluation log and per-class metrics are not present, so the repository makes no held-out accuracy claim. See [`MODEL_CARD.md`](MODEL_CARD.md).

## Repository map

```text
src/juno_sketch_tutor/
├─ scripts/drawbot_core.py            ROS-independent tested logic
├─ scripts/drawbot_vision.py          shared OpenCV preprocessing
├─ scripts/sketch_game_node.py        camera, inference, publishing and TTS
├─ scripts/voice_natural_command_node.py
├─ scripts/standalone_demo.py         synthetic preprocessing visualization
├─ scripts/train_drawing_classifier.py model inspector + optional trainer
├─ launch/                            ROS launch files
├─ models/drawing_classifier.h5
└─ test/test_drawbot_apis.py
```

## License

Project code is [MIT licensed](LICENSE). The Quick, Draw! dataset is released by Google under CC BY 4.0; see the model card for attribution and limitations.
