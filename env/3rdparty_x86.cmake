#Protobuf
find_package(Protobuf REQUIRED)
message(STATUS "Protobuf library status:")
message(STATUS "    libraries: ${PROTOBUF_LIBRARY}")
message(STATUS "    include path: ${PROTOBUF_INCLUDE_DIR}")

#Boost
find_package(Boost REQUIRED)
message(STATUS "Boost library status:")
message(STATUS "    include path: ${Boost_INCLUDE_DIRS}")
message(STATUS "    libraries: ${Boost_LIBRARIES}")
set(Boost_LIBRARIES libboost_wserialization.so libboost_wave.so
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
find_package(fmt REQUIRED)
set(FMT_LIBRARY fmt::fmt)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#MNN
if (${WITH_MNN})
    set(MnnRootPath ~/inst/MNN)
    include_directories(${MnnRootPath}/include)
    link_directories(${MnnRootPath}/build)
    set(MNN_LIBRARY MNN)
endif ()

#Tengine
if (${WITH_TENGINE})
    set(TengineRootPath ~/inst/Tengine/build/install)
    set(Tengine3rdPartyPath ~/inst/Tengine)
    include_directories(${TengineRootPath}/include)
    link_directories(${TengineRootPath}/lib)
    link_directories(${Tengine3rdPartyPath}/3rdparty/tim-vx/lib/x86_64)
    set(TENGINE_LIBRARY tengine-lite OpenVX)
endif ()