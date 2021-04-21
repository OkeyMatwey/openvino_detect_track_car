cmake_minimum_required(VERSION 3.17)

project(openvino)
set (CMAKE_PREFIX_PATH
        "c:/Program Files (x86)/Intel/openvino_2021.3.394/deployment_tools/inference_engine/"
        "c:/Program Files (x86)/Intel/openvino_2021.3.394/opencv/")

find_package(InferenceEngine)
if (InferenceEngine_FOUND)
    message(STATUS "InferenceEngine is enabled")
    message(STATUS "InferenceEngine_INCLUDE_DIRS=${InferenceEngine_INCLUDE_DIRS}")
    message(STATUS "InferenceEngine_LIBS=${InferenceEngine_LIBS}")
else()
    message(STATUS "InferenceEngine is disabled or not found,  skiped")
    return()
endif()

find_package(OpenCV)
if(OpenCV_FOUND)
    message(STATUS "OPENCV is enabled")
    message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
else()
    message(STATUS "OPENCV is disabled or not found,  skiped")
    return()
endif()

#if(UI_OUTPUT)
#    add_definitions(-DUI_OUTPUT)
#endif()

include_directories(include)

add_executable(openvino main.cpp)

target_link_libraries(openvino ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})


