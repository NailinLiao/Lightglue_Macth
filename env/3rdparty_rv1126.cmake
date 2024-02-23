SET(TOOLCHAIN_HOME "/home/wallel/cross/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf")

# this is required
#SET(CMAKE_SYSTEM_NAME Linux)

# specify the cross compiler
SET(CMAKE_C_COMPILER ${TOOLCHAIN_HOME}/bin/arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER ${TOOLCHAIN_HOME}/bin/arm-linux-gnueabihf-g++)

# where is the target environment
SET(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_HOME})

# search for programs in the build host directories (not necessary)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

add_definitions(-DUSE_NEON)

#Protobuf
include_directories(/home/wallel/cross/protobuf/install/include)
link_directories(/home/wallel/cross/protobuf/install/lib)
set(PROTOBUF_LIBRARY protobuf protobuf-lite protoc)

#JSON
include_directories(3rdparty/nlohmann)
include_directories(3rdparty)

#Boost
include_directories(/home/wallel/cross/boost/install/include)
link_directories(/home/wallel/cross/boost/install/lib)
set(BOOST_LIBRARY libboost_wserialization.so libboost_wave.so
        libboost_unit_test_framework.so libboost_type_erasure.so libboost_timer.so
        libboost_thread.so libboost_system.so libboost_serialization.so
        libboost_regex.so libboost_random.so libboost_program_options.so
        libboost_prg_exec_monitor.so libboost_nowide.so libboost_math_tr1l.so
        libboost_math_tr1f.so libboost_math_tr1.so libboost_math_c99l.so
        libboost_log_setup.so libboost_log.so libboost_locale.so
        libboost_json.so libboost_graph.so libboost_filesystem.so
        libboost_date_time.so libboost_contract.so libboost_context.so
        libboost_container.so libboost_chrono.so libboost_atomic.so)

#fmt
include_directories(/home/wallel/cross/fmt/install/include)
link_directories(/home/wallel/cross/fmt/install/lib)
set(FMT_LIBRARY fmt)

#OpenCV
include_directories(/home/wallel/cross/opencv/install/include)
include_directories(/home/wallel/cross/opencv/install/include/opencv4)
link_directories(/home/wallel/cross/opencv/install/lib)
set(OpenCV_LIBS opencv_world)

#MNN
#include_directories(/home/wallel/cross/opencv/install/include)
#link_libraries(/home/wallel/cross/opencv/install/lib)
#set(OPENCV_LIBRARY opencv_world)

#FreeType
include_directories(/home/wallel/cross/freetype-2.11.1/install/include/freetype2)
include_directories(/home/wallel/cross/freetype-2.11.1/install/include/freetype2/freetype)
link_directories(/home/wallel/cross/freetype-2.11.1/install/lib)
set(FREETYPE_LIBRARY freetype)

#RV1126
link_directories(/home/wallel/cross/target/lib)
link_directories(/home/wallel/cross/target/usr/lib)

#RGA
set(RGA_LIBRARY rga)

#RKMEDIA
add_definitions(-DRKAIQ)
include_directories(/home/wallel/cross/rkinclude/rknn)
include_directories(/home/wallel/cross/rkinclude/rkmedia)
include_directories(/home/wallel/cross/rkinclude/rkrga)
include_directories(/home/wallel/cross/rkinclude/rkrga/rga)
include_directories(/home/wallel/cross/rkinclude/rkmedia/easymedia)

include_directories(/home/wallel/cross/rkinclude/rkaiq/algos
        /home/wallel/cross/rkinclude/rkaiq/algos/a3dlut
        /home/wallel/cross/rkinclude/rkaiq/algos/ablc
        /home/wallel/cross/rkinclude/rkaiq/algos/accm
        /home/wallel/cross/rkinclude/rkaiq/algos/acp
        /home/wallel/cross/rkinclude/rkaiq/algos/adebayer
        /home/wallel/cross/rkinclude/rkaiq/algos/adehaze
        /home/wallel/cross/rkinclude/rkaiq/algos/adpcc
        /home/wallel/cross/rkinclude/rkaiq/algos/ae
        /home/wallel/cross/rkinclude/rkaiq/algos/af
        /home/wallel/cross/rkinclude/rkaiq/algos/agamma
        /home/wallel/cross/rkinclude/rkaiq/algos/ahdr
        /home/wallel/cross/rkinclude/rkaiq/algos/aie
        /home/wallel/cross/rkinclude/rkaiq/algos/alsc
        /home/wallel/cross/rkinclude/rkaiq/algos/anr
        /home/wallel/cross/rkinclude/rkaiq/algos/aorb
        /home/wallel/cross/rkinclude/rkaiq/algos/asd
        /home/wallel/cross/rkinclude/rkaiq/algos/asharp
        /home/wallel/cross/rkinclude/rkaiq/algos/awb
        /home/wallel/cross/rkinclude/rkaiq/common
        /home/wallel/cross/rkinclude/rkaiq/iq_parser
        /home/wallel/cross/rkinclude/rkaiq/uAPI
        /home/wallel/cross/rkinclude/rkaiq/xcore)

include_directories(/home/wallel/cross/librtsp/include)
link_directories(/home/wallel/cross/librtsp/lib)

#RKNN
include_directories(/home/wallel/cross/rknpu/include)
link_directories(/home/wallel/cross/rknpu/lib)

set(RKMEDIA_LIBRARY drm rockchip_mpp liveMedia groupsock BasicUsageEnvironment UsageEnvironment
        easymedia rkaiq rtsp asound RKAP_3A RKAP_ANR RKAP_Common v4l2 jpeg v4lconvert rockx od_share
        md_share sqlite3 rockface rknn_runtime rknn_api OpenVX ArchModelSw CLC GAL
        neuralnetworks NNArchPerf NNGPUBinary NNVXCBinary OpenVX OpenVXU
        Ovx12VXCBinary rknn_utils VSC ann_plugin rknn_plugin dl)

set(RKNN_LIBRARY rknn_api rknn_runtime OpenVX ArchModelSw CLC GAL
        neuralnetworks NNArchPerf NNGPUBinary NNVXCBinary OpenVX OpenVXU
        Ovx12VXCBinary rknn_utils VSC ann_plugin rknn_plugin dl)

#FFTW
include_directories(/home/wallel/cross/fftw-3.3.10/install/include)
link_directories(/home/wallel/cross/fftw-3.3.10/install/lib)
set(FFTW_LIBRARY fftw3)

#Eigen
include_directories(/home/wallel/cross/eigen-3.4.0/)