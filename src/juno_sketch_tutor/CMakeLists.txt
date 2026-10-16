cmake_minimum_required(VERSION 3.0.2)
project(juno_sketch_tutor)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS cv_bridge rospy sensor_msgs std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/drawbot_core.py
  scripts/drawbot_vision.py
  scripts/sketch_game_node.py
  scripts/voice_natural_command_node.py
  scripts/train_drawing_classifier.py
  scripts/standalone_demo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch models docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_drawbot_apis.py)
endif()
