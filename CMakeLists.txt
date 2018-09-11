CMAKE_MINIMUM_REQUIRED(VERSION 3.4)
PROJECT(SIMDwrap)

ADD_LIBRARY(SIMDwrap INTERFACE)
TARGET_INCLUDE_DIRECTORIES(SIMDwrap INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SIMDwrap>
)

IF(MSVC)
    TARGET_COMPILE_OPTIONS(SIMDwrap INTERFACE "/arch:AVX2" "/std:c++17")
    # interface libraries can't use target_properties for CXX standard, so 
    # need to eventually expand this for clang/gcc etc
ENDIF()

SET(simd_wrap_srcs 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/simd_traits.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/storage_proxy.hpp"
)

ADD_CUSTOM_TARGET(SIMDwrap.sources SOURCES ${simd_wrap_srcs})

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SIMDwrap/")

OPTION(BUILD_TESTING OFF "Build tests for the vector classes and objects")

IF(BUILD_TESTING) 
    ADD_EXECUTABLE(type_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/types.cpp")
    TARGET_LINK_LIBRARIES(type_tests PRIVATE SIMDwrap)
ENDIF()
