cmake_minimum_required(VERSION 3.16)
project(inference_tflite C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(TF_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)

get_filename_component(TF_SOURCE_DIR
  "~/tensorflow/" # change path to your tensorflow directory (clone the git repository at https://github.com/tensorflow/tensorflow, and checkout to desired release)
  ABSOLUTE
)

add_subdirectory(
  "${TF_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 17)
add_executable(inference_tflite
  inference_tflite.cpp
)

target_link_libraries(inference_tflite ${OpenCV_LIBS})
target_link_libraries(inference_tflite
  tensorflow-lite
)
