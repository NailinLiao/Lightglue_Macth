cmake_minimum_required(VERSION 3.18)
project(lightglue_Macth)

set(CMAKE_CXX_STANDARD 17)

# 添加OpenCV查找模块
find_package(OpenCV REQUIRED core highgui imgproc features2d videoio imgcodecs calib3d)

if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found.")
endif ()

include(env/3rdparty_rk3588.cmake)


include_directories(include ${OpenCV_INCLUDE_DIRS})

# 收集源文件
file(GLOB_RECURSE ORB_SRC src/*.cpp src/*.cc)
file(GLOB_RECURSE ORB_HEAD include/*.hpp include/*.h)

#add_executable( main.cpp src/ORB_WORKER.cpp)

add_executable(lightglue_Macth
        ${ORB_SRC}
        ${ORB_HEAD}
        ${CORE_LIB_SRC}
        ${CORE_LIB_HEAD}
        ${OPENCV_LIB_SRC}
        ${OPENCV_LIB_HEAD}
        main.cpp
        )

target_link_libraries(lightglue_Macth
        pthread
        ${FMT_LIBRARY}
        ${BOOST_LIBRARY}
        ${RGA_LIBRARY}
        ${OpenCV_LIBS}
        ${RKNN_LIBRARY}
        ${FFTW_LIBRARY}
        ${FREETYPE_LIBRARY}
        ${MPP_LIBRARY}
        ${MPPENCODER_LIBRARY}
        ${RKAIQ_LIBRARY}
        ${DB_LIBRARY}
        )

## 链接OpenCV库到目标可执行文件
#target_link_libraries(ORB_Jitter PRIVATE ${OpenCV_LIBS})