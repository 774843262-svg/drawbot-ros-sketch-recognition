# `juno_sketch_tutor` ROS package

See the [repository README](../../README.md) for architecture, safety defaults and offline tests.

## Target environment

- Ubuntu 20.04
- ROS Noetic (legacy/EOL)
- `usb_cam`, `image_view`, `cv_bridge`, `rospy`, `sensor_msgs`, `std_msgs`
- TensorFlow, OpenCV, NumPy and eSpeak for the full robot path

## Build and launch

```bash
catkin_make
source devel/setup.bash
roslaunch juno_sketch_tutor juno_sketch_game_usb.launch
```

The voice launch uses Google's speech-recognition service and therefore stays disabled by default:

```bash
# No audio upload; voice node idles
roslaunch juno_sketch_tutor juno_sketch_game_voice.launch

# Explicit cloud-audio opt-in
roslaunch juno_sketch_tutor juno_sketch_game_voice.launch allow_cloud_stt:=true
```

Set `debug_dir:=/path/you/control` only when saving camera frames is appropriate and consented to. With the default empty value, no raw or processed frames are written.
