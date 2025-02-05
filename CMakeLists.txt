cmake_minimum_required(VERSION 3.0.2)
project(rofunc_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# Require c++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "-std=c++11")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  actionlib_msgs
  geometry_msgs
  nav_msgs
  tf
  dynamic_reconfigure
  message_generation
  rosunit
  sound_play
)



## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY VoiceQA/speech2text/msg
  FILES
  speech_recognition.msg
)

## Generate services in the 'srv' folder
add_service_files(
  DIRECTORY VoiceQA/text2speech/srv
  FILES 
  Synthesizer.srv 
  Polly.srv
)

## Generate actions in the 'action' folder
add_action_files(
  DIRECTORY VoiceQA/text2speech/action
  FILES
  Speech.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES 
  actionlib_msgs 
  std_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  OptiLocation/cfg/MocapOptitrack.cfg
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  LIBRARIES rofunc_ros
  INCLUDE_DIRS OptiLocation/include
  CATKIN_DEPENDS roscppactionlib_msgs message_runtime rospy std_msgs
  dynamic_reconfigure
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  OptiLocation/include
  ${catkin_INCLUDE_DIRS}
)

add_subdirectory(OptiLocation/src)


## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
install(PROGRAMS
  VoiceQA/text2speech/scripts/polly_node.py
  VoiceQA/text2speech/scripts/synthesizer_node.py
  VoiceQA/text2speech/scripts/tts_node.py
  VoiceQA/text2speech/scripts/voicer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  ## Add folders to be run by python nosetests
  catkin_add_nosetests(VoiceQA/text2speech/test/test_unit_synthesizer.py)
  catkin_add_nosetests(VoiceQA/text2speech/test/test_unit_polly.py)
  
  if(BUILD_AWS_TESTING)
      find_package(rostest REQUIRED COMPONENTS tts)
      add_rostest(test/integration_tests.test DEPENDENCIES ${tts_EXPORTED_TARGETS})
  endif()
endif()

