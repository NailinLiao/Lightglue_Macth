# this is required
SET(CMAKE_SYSTEM_NAME Linux)

# search for programs in the build host directories (not necessary)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

add_definitions(-DUSE_NEON)

# Protobuf
# include_directories(/home/linaro/inst/boost-1.81.0/include)
# include_directories(/home/linaro/Desktop/env/opencv_new/opencv/3rdparty/protobuf/src)

# link_directories(/home/linaro/Desktop/env/protobuf-3.20.x/output/lib)
set(PROTOBUF_LIBRARY protobuf protobuf-lite protoc)

# Boost
include_directories(/home/linaro/inst/boost-1.81.0/include)

# include_directories(/home/linaro/Desktop/env/boost_1_81_0)
link_directories(/home/linaro/inst/boost-1.81.0/lib)
set(BOOST_LIBRARY libboost_wserialization.so libboost_wave.so
        libboost_unit_test_framework.so libboost_type_erasure.so libboost_timer.so
        libboost_thread.so libboost_system.so libboost_serialization.so
        libboost_regex.so libboost_random.so libboost_program_options.so
        libboost_prg_exec_monitor.so libboost_nowide.so libboost_math_tr1l.so
        libboost_math_tr1f.so libboost_math_tr1.so libboost_math_c99l.so
        libboost_log.so libboost_locale.so
        libboost_json.so libboost_graph.so libboost_filesystem.so
        libboost_date_time.so libboost_contract.so libboost_context.so
        libboost_container.so libboost_chrono.so libboost_atomic.so)

# OpenCV
include_directories(/home/linaro/inst/opencv-4.5.5/include)
include_directories(/home/linaro/inst/opencv-4.5.5/include/opencv4)
link_directories(/home/linaro/inst/opencv-4.5.5/lib)
set(OpenCV_LIBS libopencv_core.so libopencv_features2d.so libopencv_highgui.so libopencv_imgcodecs.so
        libopencv_imgproc.so libopencv_videoio.so libopencv_video.so libopencv_calib3d.so
        libopencv_dnn.so libopencv_flann.so libopencv_gapi.so libopencv_ml.so libopencv_objdetect.so
        libopencv_photo.so libopencv_stitching.so)

# RGA
include_directories(/usr/include/rga)
set(RGA_LIBRARY rga)

# ffmpeg 4.4
link_directories(/home/linaro/inst/ffmpeg-4.4/lib/)
include_directories(/home/linaro/inst/ffmpeg-4.4/include/)

# RKNN
# include_directories(/home/linaro/dev/unitTest/3rdparty)
include_directories(/home/linaro/inst/rknn)
set(RKNN_LIBRARY rknnrt)

# link_directories(/home/wallel/cross/rknpu/lib)

# mpp
include_directories(/opt/mpp/inc)
include_directories(/opt/mpp/osal/inc)
include_directories(/opt/mpp/mpp/codec/inc)
link_directories(/opt/mpp/build/linux/aarch64/mpp)
set(MPP_LIBRARY rockchip_mpp swscale avutil avcodec avformat)

# eigen
include_directories(/home/linaro/inst/eigen-3.4/include/eigen3)


#json
include_directories("/usr/include/jsoncpp")
# 添加JsonCpp头文件到特定目标
include_directories(${JsonCpp_INCLUDE_DIRS})

# 添加JsonCpp头文件到特定目标
#target_include_directories(test PRIVATE ${JsonCpp_INCLUDE_DIRS})

#rkaiq
link_directories(/home/linaro/inst/rkaiq/)
include_directories(/home/linaro/inst/rkaiq/)
include_directories(/home/linaro/inst/rkaiq/algos)
include_directories(/home/linaro/inst/rkaiq/common)
include_directories(/home/linaro/inst/rkaiq/iq_parser)
include_directories(/home/linaro/inst/rkaiq/iq_parser_v2)
include_directories(/home/linaro/inst/rkaiq/xcore)
include_directories(/home/linaro/inst/rkaiq/uAPI2)
set(RKAIQ_LIBRARY rkaiq dl)

#freetype2
link_directories(/home/linaro/inst/freetype/install/lib)
include_directories(/home/linaro/inst/freetype/install/include/freetype2/freetype)
include_directories(/home/linaro/inst/freetype/install/include/freetype2)
set(FREETYPE_LIBRARY freetype)

#fmt
include_directories(/home/linaro/inst/fmt/include)
link_directories(/home/linaro/inst/fmt/build)
set(FMT_LIBRARY fmt)

#db
include_directories(/home/linaro/inst/db/install/include)
link_directories(/home/linaro/inst/db/install/lib)
set(DB_LIBRARY db db_cxx)